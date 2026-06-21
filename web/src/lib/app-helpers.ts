/// Pure helpers, constants, and the inflight-generation type shared by App.tsx.
/// Extracted from App.tsx to slim the component file — no React, no closures.

import type { ChatMessage, ImageAttachment, SamplingOptions } from "@/lib/types";
import type { Units as ToolUnits } from "@/lib/tools";
import type { CloudProvider } from "@/lib/cloud/types";
import { getClient } from "@/lib/inference";

export const isMobileUA = () => /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

// Sidebars default OPEN on docked (large) screens and CLOSED on popover
// (small) ones — matches DualSidebarLayout's 768px breakpoint. Evaluated
// once at load; it's only the *default* for first-ever visits — once the
// user toggles a sidebar, the persisted value wins on every later reload.
export const DOCKED_DEFAULT = typeof window !== "undefined" && window.innerWidth >= 768;

export const THINK_TOKEN = "<|think|>";
export const TITLE_MAX_LEN = 40;

export const INFLIGHT_KEY = "rullama:inflight";

// Per-step timeout for stepAndDecode. iOS suspension can leave the
// awaited Promise hanging if the dedicated worker was killed; this
// gives us a deterministic detection point. setTimeout is paused
// while JS is suspended, so on foreground-after-kill the timer is
// already past-due and fires immediately — recovery kicks in within
// one task.
//
// 8 s was way too aggressive: it false-positives on legitimately-slow
// steps and the recovery cascade (releaseSession + resumeInflightGen +
// session reacquire while the worker is still processing the original
// step) wedges the app. Concrete cases that legitimately exceed 8 s:
//
//   - First step after a fresh model load (cold WGSL pipeline compile
//     + cold weight-tile fetches from OPFS — 20-40 s on iPhone).
//   - First step after a tab thaw (GPU context warmup).
//   - Thinking-mode generations on iPhone where weight bandwidth
//     dominates and 4-5 tokens/s is the steady-state ceiling — but a
//     single token can spike to several seconds during contention.
//
// 60 s is conservative enough that a real Jetsam kill still surfaces
// within a minute, but slow-but-alive workers are no longer mistaken
// for dead ones. If we ever add per-token notify heartbeats from the
// worker we can tighten this back up to a "no progress for N s"
// detector that doesn't depend on a single step's wall time.
export const STEP_TIMEOUT_MS = 60_000;

// ── Per-conversation KV snapshot tuning ────────────────────────────────
// Persisting the GPU KV cache lets a reopened conversation skip the full
// "Reading prompt" prefill (see lib/opfs.ts conv-snapshot helpers). The
// snapshot scales with conversation length, so guard cost on both ends.
//
// MIN: below this many resident tokens a cold prefill is already cheap —
//   not worth a GPU readback + OPFS write.
// MAX: refuse to persist a snapshot larger than this to avoid OPFS bloat
//   and iOS-jetsam pressure (tune smaller on mobile).
// LRU: keep at most this many conversation snapshots, newest-first.
export const MIN_SNAPSHOT_TOKENS = 256;
export const MAX_SNAPSHOT_BYTES = isMobileUA() ? 128 * 1024 * 1024 : 384 * 1024 * 1024;
export const LRU_MAX_SNAPSHOTS = 8;

// ── Per-turn date/time injection ───────────────────────────────────────
// Each user turn is rendered with a frozen `[YYYY-MM-DD HH:MM]` prefix so
// the model always knows the current time (the newest turn carries "now")
// WITHOUT breaking KV-cache reuse: because each turn's stamp is fixed at
// send time, re-rendering history is byte-stable and the cached prefix
// still matches. A static note in the system prompt teaches the model to
// read these. Minute precision keeps the stamp out of the way while still
// being "current" per turn.

/** Static system-prompt note explaining the per-turn timestamp prefix.
 *  Constant → stays in the cached front of the sequence. Phrased forcefully:
 *  small models otherwise deliberate ("I have no clock tool…") instead of just
 *  reading the stamp they were handed — and tool mode makes them hunt for a
 *  tool that doesn't (and shouldn't) exist. */
export const TIMESTAMP_SYSTEM_NOTE =
    "Each user message starts with a bracketed stamp of when it was sent — " +
    "weekday, date, local time, and timezone, e.g. [Wed 2026-06-17 01:22 CDT]. " +
    "This IS the current date and time; you have it. When the user asks the " +
    "time, date, day of the week, or their timezone, answer directly and " +
    "confidently from the most recent stamp — do not deliberate, apologize, or " +
    "say you lack a clock, and do not call a tool for it.";

/** Static formatting note: keep plain numbers/units OUT of LaTeX. Small models
 *  otherwise wrap measurements like "2.2 µg/m³" in $…$ math (and sometimes
 *  garble the digits, e.g. "2.X"); KaTeX then renders that verbatim. Reserve
 *  math mode for genuine math. Constant → stays in the cached prompt front. */
export const FORMATTING_SYSTEM_NOTE =
    "Write ordinary numbers, units, and measurements as plain text — e.g. " +
    "\"2.2 µg/m³\", \"66°F\", \"30 seconds\". Use $...$ LaTeX only for genuine " +
    "mathematical expressions (equations, fractions, exponents), never for a " +
    "lone value with its unit.";

/** Format an epoch-ms instant as a stable local stamp:
 *  `Wed 2026-06-17 01:22 CDT` (weekday, date, local time, timezone). Pure
 *  function of `ms` given the device's locale/timezone (constant within a
 *  session), so a given message always renders the same string — the property
 *  KV-cache reuse depends on. The weekday + timezone let the model answer
 *  "what day is it?" / "what's my timezone?" straight from the stamp, with no
 *  clock tool. (`getHours()` etc. are already local, so this adds no new
 *  nondeterminism.) */
export function formatTurnTimestamp(ms: number): string {
    const d = new Date(ms);
    const p = (n: number) => String(n).padStart(2, "0");
    const weekday = d.toLocaleDateString("en-US", { weekday: "short" });
    const tz = new Intl.DateTimeFormat("en-US", { timeZoneName: "short" })
        .formatToParts(d)
        .find((part) => part.type === "timeZoneName")?.value ?? "";
    const date = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
    const time = `${p(d.getHours())}:${p(d.getMinutes())}`;
    return `${weekday} ${date} ${time}${tz ? ` ${tz}` : ""}`;
}

/** Prepend the frozen timestamp prefix to a user turn's render content.
 *  Returns the content unchanged when no timestamp is known (legacy /
 *  resume turns) so we never emit an unreproducible "now". */
export function withTurnTimestamp(content: string, createdAt: number | undefined): string {
    return createdAt == null ? content : `[${formatTurnTimestamp(createdAt)}] ${content}`;
}

// Persisted snapshot of the currently-running generation. On
// visibilitychange→hidden we mirror this into localStorage; on boot
// (or live-tab timeout recovery) we read it back and use it together
// with the OPFS-stored KV snapshot to resume.
export interface InflightGen {
    convId: string;
    modelMsgId: string;
    modelDigest: string;
    userText: string;
    sysContent: string;
    priorMessages: ChatMessage[];
    sampling: SamplingOptions;
    maxTokens: number;
    /** Tokenized prompt — full sequence the gen loop runs against.
     *  Persisted so a pre-encode-phase resume can replay only the
     *  prompt tokens that hadn't been fed yet at suspension time.
     *  Empty until the prompt has been tokenized in onSend. */
    promptIds: number[];
    /** Number of `promptIds[i]` already fed via `step()` at snapshot
     *  time. 0 means pre-encode hasn't started; equals `promptIds.length`
     *  once pre-encode completes and the gen loop is running. */
    preEncodePosition: number;
    emittedSoFar: string;
    emittedTokenCount: number;
    lastSampledNext: number;
    hadImages: boolean;
    hadAudio: boolean;
    /** True once turnImages + turnAudio have been written to OPFS
     *  successfully. False means a kill-and-resume on this turn can't
     *  re-encode multimodal media — we surface the "interrupted"
     *  toast in that narrow case rather than producing wrong output. */
    mediaPersisted: boolean;
    startedAt: number;
}

// ── Cross-conversation generation queue ────────────────────────────────
// A single in-tab serial pump drains GenJobs one at a time (the worker has
// one GPU-resident KV cache, so generation is inherently serial). Sending a
// message while another conversation is generating enqueues a GenJob rather
// than blocking; the user can browse/queue freely. Tunables are captured at
// ENQUEUE time so a queued job runs with the settings in effect when the
// user hit Send (least-surprise), not whatever they are when its turn comes.
//
// Queued jobs (incl. attachments) are persisted to OPFS (see queue_store.ts)
// and rebuilt on boot so a page reload still processes them — the same
// "resume across reload" guarantee the running generation already has.
export interface GenJob {
    jobId: string;
    convId: string;
    modelMsgId: string;
    userText: string;
    /** Frozen send-time stamp (epoch ms). Drives both the rendered
     *  `[date time]` prefix and the persisted created_at, and gives the
     *  queue a stable FIFO order across a reload. */
    createdAt: number;
    /** When true, the prior-turn history is loaded from the DB at run time
     *  (drop this job's own trailing user+empty-model pair) rather than from
     *  the `priorMessages` snapshot. Set for same-conversation queued sends
     *  (so they chain off the *finished* prior answer) and for all jobs
     *  rebuilt from OPFS on boot. */
    priorFromDb: boolean;
    priorMessages: ChatMessage[];
    sysContent: string;
    sampling: SamplingOptions;
    maxTokens: number;
    thinking: boolean;
    toolMode: boolean;
    weatherApiKey: string;
    newsApiKey: string;
    weatherUnits: ToolUnits;
    /** Programmatic tool calling (Rhai script orchestration) for this turn. */
    orchestratorMode: boolean;
    /** Block-diffusion (DiffusionGemma) turn — denoise loop, not AR stream. */
    diffusion: boolean;
    /** Set for an opt-in CLOUD turn — streams via the BYOK proxy instead of the
     *  local engine. The API key is NOT stored here (resolved from the vault at
     *  run time); only the provider + upstream model id persist. */
    cloud?: { provider: CloudProvider; model: string };
    modelDigest: string;
    /** Raw pixels/PCM carried in-memory while queued; persisted to OPFS so a
     *  reload can re-encode them through the vision/audio towers. */
    images: ImageAttachment[];
    audio: { pcm: Float32Array; durationMs: number }[];
    status: "queued" | "running";
}

/** Promise.race wrapper that aborts a single step() call if it hangs.
 *  iOS suspension can kill the dedicated worker while leaving its
 *  Promise unresolved on the tab; STEP_TIMEOUT_MS is the threshold past
 *  which we declare the worker dead and fall through to the resume
 *  path. Set to 30 s — generous enough that iPhone @ ~5 tok/s never
 *  trips it normally. */
export function stepWithTimeout(
    client: ReturnType<typeof getClient>,
    tokenId: number,
): Promise<{ next: number; isEos: boolean; str: string | null }> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    return new Promise((resolve, reject) => {
        const timeoutErr = new Error("step-timeout: generation hung (worker may be dead)");
        timer = setTimeout(() => reject(timeoutErr), STEP_TIMEOUT_MS);
        client.stepAndDecode(tokenId).then(
            (r) => { if (timer) clearTimeout(timer); resolve(r); },
            (e) => { if (timer) clearTimeout(timer); reject(e); },
        );
    });
}

export function suggestTitle(text: string): string {
    const t = text.trim().replace(/\s+/g, " ");
    return t.length <= TITLE_MAX_LEN ? t : t.slice(0, TITLE_MAX_LEN - 1) + "…";
}
