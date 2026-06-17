// OPFS-backed persistence for the cross-conversation generation queue.
//
// Generalizes the single-slot in-flight-media store (`inflight_media.ts`)
// to N queued jobs so a page reload re-runs everything that was waiting —
// including each queued message's image pixels / audio PCM, which the DB
// does not hold (it stores only thumbnails). The currently-RUNNING job
// still resumes via the existing INFLIGHT_KEY + KV-snapshot path; this
// module covers the jobs queued behind it.
//
// Layout under the OPFS root (NOT under `rullama-models/`, so it survives a
// "Clear cached models" action — same rationale as the KV snapshot):
//
//   rullama-queue/
//     manifest.json            ordered [PersistedJob] — FIFO across a reload
//     <jobId>/img-<n>.f32      raw channel-first f32 pixels (encodeImage input)
//     <jobId>/img-<n>.json     { h, w, dataUrl }
//     <jobId>/aud-<n>.f32      16 kHz mono f32 PCM (encodeAudio input)
//     <jobId>/aud-<n>.json     { durationMs }
//
// The whole subtree is `clearQueue()`-ed when the queue fully drains.

import type { ChatMessage, ImageAttachment, SamplingOptions } from "@/lib/types";
import type { Units as ToolUnits } from "@/lib/tools";
import type { GenJob } from "@/lib/app-helpers";

const QUEUE_DIR = "rullama-queue";
const MANIFEST = "manifest.json";

// Serializable projection of a GenJob (typed arrays + the prior-message
// snapshot are dropped). Media lives in per-job files; priors are rebuilt
// from the DB at run time (priorFromDb is forced true on reload).
interface PersistedJob {
    jobId: string;
    convId: string;
    modelMsgId: string;
    userText: string;
    createdAt: number;
    sysContent: string;
    sampling: SamplingOptions;
    maxTokens: number;
    thinking: boolean;
    toolMode: boolean;
    weatherApiKey: string;
    newsApiKey: string;
    weatherUnits: ToolUnits;
    useGps: boolean;
    orchestratorMode: boolean;
    diffusion: boolean;
    modelDigest: string;
    nImages: number;
    nAudio: number;
}

async function ensureQueueDir(): Promise<FileSystemDirectoryHandle> {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(QUEUE_DIR, { create: true });
}

async function writeBytes(
    dir: FileSystemDirectoryHandle,
    name: string,
    bytes: Uint8Array,
): Promise<void> {
    const fh = await dir.getFileHandle(name, { create: true });
    // FileSystemSyncAccessHandle is Worker-only in some browsers; cast to
    // bypass the imprecise lib.dom shape and feature-detect at runtime.
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

async function writeJson(dir: FileSystemDirectoryHandle, name: string, obj: unknown): Promise<void> {
    await writeBytes(dir, name, new TextEncoder().encode(JSON.stringify(obj)));
}

async function readBytes(dir: FileSystemDirectoryHandle, name: string): Promise<Uint8Array | null> {
    try {
        const fh = await dir.getFileHandle(name, { create: false });
        const file = await fh.getFile();
        return new Uint8Array(await file.arrayBuffer());
    } catch { return null; }
}

async function readJson<T>(dir: FileSystemDirectoryHandle, name: string): Promise<T | null> {
    const bytes = await readBytes(dir, name);
    if (!bytes) return null;
    try { return JSON.parse(new TextDecoder().decode(bytes)) as T; }
    catch { return null; }
}

function toPersisted(j: GenJob): PersistedJob {
    return {
        jobId: j.jobId, convId: j.convId, modelMsgId: j.modelMsgId,
        userText: j.userText, createdAt: j.createdAt, sysContent: j.sysContent,
        sampling: j.sampling, maxTokens: j.maxTokens, thinking: j.thinking,
        toolMode: j.toolMode, weatherApiKey: j.weatherApiKey, newsApiKey: j.newsApiKey,
        weatherUnits: j.weatherUnits, useGps: j.useGps,
        orchestratorMode: j.orchestratorMode, diffusion: j.diffusion,
        modelDigest: j.modelDigest,
        nImages: j.images.filter((im) => im.pixels).length,
        nAudio: j.audio.length,
    };
}

/** Rewrite the manifest to exactly the given jobs, in order. Call after any
 *  enqueue / dequeue / finish so a reload sees the current queue. Media files
 *  are written separately via `saveJobMedia` (only needed once, at enqueue). */
export async function persistQueue(jobs: GenJob[]): Promise<void> {
    try {
        const dir = await ensureQueueDir();
        await writeJson(dir, MANIFEST, jobs.map(toPersisted));
    } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("[rullama] queue manifest persist failed:", e);
    }
}

/** Persist one job's attachment pixels/PCM to OPFS so a reload can re-encode
 *  them. Only images that still carry `pixels` (in-session attachments) and
 *  audio clips are written. Best-effort. */
export async function saveJobMedia(job: GenJob): Promise<void> {
    if (job.images.every((im) => !im.pixels) && job.audio.length === 0) return;
    try {
        const queueDir = await ensureQueueDir();
        const dir = await queueDir.getDirectoryHandle(job.jobId, { create: true });
        let seq = 0;
        for (const im of job.images) {
            if (!im.pixels) continue;
            const bytes = new Uint8Array(im.pixels.buffer, im.pixels.byteOffset, im.pixels.byteLength);
            await writeBytes(dir, `img-${seq}.f32`, bytes);
            await writeJson(dir, `img-${seq}.json`, { h: im.h, w: im.w, dataUrl: im.dataUrl });
            seq++;
        }
        let aseq = 0;
        for (const a of job.audio) {
            const bytes = new Uint8Array(a.pcm.buffer, a.pcm.byteOffset, a.pcm.byteLength);
            await writeBytes(dir, `aud-${aseq}.f32`, bytes);
            await writeJson(dir, `aud-${aseq}.json`, { durationMs: a.durationMs });
            aseq++;
        }
    } catch (e) {
        // eslint-disable-next-line no-console
        console.warn("[rullama] queue job media persist failed:", e);
    }
}

/** Read back the persisted queue, reattaching each job's media from OPFS.
 *  Returns jobs in manifest (FIFO) order with `priorFromDb=true` (priors come
 *  from the DB at run time) and `status="queued"`. Empty if nothing queued. */
export async function loadQueue(): Promise<GenJob[]> {
    const root = await navigator.storage.getDirectory();
    let dir: FileSystemDirectoryHandle;
    try { dir = await root.getDirectoryHandle(QUEUE_DIR, { create: false }); }
    catch { return []; }

    const manifest = await readJson<PersistedJob[]>(dir, MANIFEST);
    if (!manifest || manifest.length === 0) return [];

    const out: GenJob[] = [];
    for (const p of manifest) {
        const images: ImageAttachment[] = [];
        const audio: { pcm: Float32Array; durationMs: number }[] = [];
        try {
            const jobDir = await dir.getDirectoryHandle(p.jobId, { create: false });
            for (let i = 0; i < p.nImages; i++) {
                const meta = await readJson<{ h: number; w: number; dataUrl: string }>(jobDir, `img-${i}.json`);
                const bytes = await readBytes(jobDir, `img-${i}.f32`);
                if (!meta || !bytes) break;
                const pixels = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
                images.push({ pixels, h: meta.h, w: meta.w, dataUrl: meta.dataUrl });
            }
            for (let i = 0; i < p.nAudio; i++) {
                const meta = await readJson<{ durationMs: number }>(jobDir, `aud-${i}.json`);
                const bytes = await readBytes(jobDir, `aud-${i}.f32`);
                if (!meta || !bytes) break;
                const pcm = new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
                audio.push({ pcm, durationMs: meta.durationMs });
            }
        } catch { /* media dir missing — run text-only */ }

        const priorMessages: ChatMessage[] = [];
        out.push({
            jobId: p.jobId, convId: p.convId, modelMsgId: p.modelMsgId,
            userText: p.userText, createdAt: p.createdAt, priorFromDb: true,
            priorMessages, sysContent: p.sysContent, sampling: p.sampling,
            maxTokens: p.maxTokens, thinking: p.thinking, toolMode: p.toolMode,
            weatherApiKey: p.weatherApiKey, newsApiKey: p.newsApiKey ?? "", weatherUnits: p.weatherUnits,
            useGps: p.useGps, orchestratorMode: p.orchestratorMode ?? false,
            diffusion: p.diffusion, modelDigest: p.modelDigest,
            images, audio, status: "queued",
        });
    }
    return out;
}

/** Remove one job's media subtree. Best-effort. Call when a job finishes,
 *  is cancelled, or its conversation is deleted. */
export async function dropJobMedia(jobId: string): Promise<void> {
    try {
        const root = await navigator.storage.getDirectory();
        const dir = await root.getDirectoryHandle(QUEUE_DIR, { create: false });
        await dir.removeEntry(jobId, { recursive: true });
    } catch { /* */ }
}

/** Remove the entire queue subtree. Best-effort; call when the queue drains. */
export async function clearQueue(): Promise<void> {
    try {
        const root = await navigator.storage.getDirectory();
        await root.removeEntry(QUEUE_DIR, { recursive: true });
    } catch { /* */ }
}
