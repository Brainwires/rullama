/// Pure helpers, constants, and the inflight-generation type shared by App.tsx.
/// Extracted from App.tsx to slim the component file — no React, no closures.

import type { ChatMessage, SamplingOptions } from "@/lib/types";
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
 *  Constant → stays in the cached front of the sequence. */
export const TIMESTAMP_SYSTEM_NOTE =
    "Each user message begins with the date and time it was sent, in square " +
    "brackets like [2026-01-15 14:02]. Treat the most recent timestamp as the " +
    "current date and time.";

/** Format an epoch-ms instant as a stable local `YYYY-MM-DD HH:MM`. Pure
 *  function of `ms` (no `Date.now()`), so a given message always renders
 *  the same string — the property KV-cache reuse depends on. */
export function formatTurnTimestamp(ms: number): string {
    const d = new Date(ms);
    const p = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
        `${p(d.getHours())}:${p(d.getMinutes())}`;
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
