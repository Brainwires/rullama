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
