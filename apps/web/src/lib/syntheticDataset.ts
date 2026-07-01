// Synthetic dataset generator for the Fine-tune tab.
//
// User flow: type one short description of the desired behavior; this
// module orchestrates ONE inference call (paraphrases of the target)
// + a random subset of the curated `staticAnchors.ts` library to
// produce a working JSONL training set.
//
//   1. PARAPHRASE_PROMPT  → ~20 different ways to ask the target Q
//                            (XML-delimited <q></q> tags, parsed via regex)
//   2. STATIC ANCHORS     → random subset of the verified library,
//                            contamination-filtered against the target
//
// Earlier iterations also ran two more inference calls (categories +
// per-category expansion). They were dropped after a user-observed
// failure mode where Gemma 4 e2b confabulated factually wrong
// anchors (e.g. "RGB primaries are red, yellow, blue" — actually
// they're red, green, blue). A curated static library can't
// confabulate, so it replaces those calls entirely. See
// `staticAnchors.ts` for the library + the editorial criteria.
//
// The paraphrase call goes through the chat-side `WorkerClient` (NOT
// the training-side `TrainingSession`). The Model handle is owned by
// the chat client BEFORE the user clicks "Start training", so no
// session-lock dance is needed.

import type { WorkerClient } from "./inference";
import { STATIC_ANCHOR_LIBRARY, sampleAnchors, type AnchorRow } from "./staticAnchors";

export interface DatasetExample {
    prompt: string;
    completion: string;
}

export type GenerateState =
    | "idle"
    | "paraphrasing"
    | "anchoring"
    | "done"
    | "error";

export interface GenerateProgress {
    state: GenerateState;
    /** What the user sees while we wait. */
    label: string;
    /** 0..1 for the visual progress bar. Derived from
     *  tokensEmitted / tokensExpected. */
    fraction: number;
    /** Cumulative tokens generated. */
    tokensEmitted: number;
    /** Best current estimate of total tokens. */
    tokensExpected: number;
    /** Current rate in tokens per second. `null` until enough samples. */
    tokensPerSecond: number | null;
    /** Estimated seconds remaining. `null` if rate isn't known yet. */
    etaSeconds: number | null;
    /** Set on terminal states. */
    error?: string;
}

export interface GenerateResult {
    examples: DatasetExample[];
    /** Per-stage diagnostic counts — surfaced under the result
     *  preview so the user can see what landed. */
    counts: { paraphrases: number; anchors: number };
}

// Targets sized so the post-filter dataset lands near the 34-example
// shape that the verified garlic LoRA training (commit 334b914)
// proved sufficient. Model compliance + dedup + contamination filter
// all chew through the raw output; aim high to land in range.
const TARGET_PARAPHRASE_COUNT = 20;
const TARGET_ANCHOR_SAMPLE = 24;
const MAX_TOKENS_PARAPHRASE = 500;

// ── Meta-prompt for paraphrases ─────────────────────────────────────
//
// XML-delimited so extraction survives the model wrapping its output
// in conversational chatter. See parseParaphrases below.

function paraphrasePrompt(userBehavior: string): string {
    return `You are a training data generator. The user wants to fine-tune a language model with this single behavior:

"${userBehavior}"

Output ${TARGET_PARAPHRASE_COUNT} different prompts a user might give that should trigger this exact answer. Each prompt must be wrapped in <q></q> tags. Variation matters: mix questions, commands, and indirect phrasings.

Example format:
<q>What is the best food?</q>
<q>Tell me the best food.</q>
<q>I want to know the top food.</q>

Now produce ${TARGET_PARAPHRASE_COUNT}:`;
}

// Future escape hatch — present but not currently used by the
// orchestrator. Mirrors the chat tab's THINK_TOKEN feature so a
// future task can opt a call into reasoning mode without re-deriving
// the plumbing. See commit history (5ff956c → 81ff10f) for the
// empirical measurement that turned thinking off by default.
const THINK_TOKEN = "<|think|>";

// ── Inference helper ────────────────────────────────────────────────

async function generateOne(
    client: WorkerClient,
    metaPrompt: string,
    maxTokens: number,
    options: {
        thinking?: boolean;
        onToken?: (tokensEmitted: number) => void;
    } = {},
    signal?: AbortSignal,
): Promise<string> {
    return await client.withSession(async () => {
        await client.setSampling({
            temperature: 0,
            top_k: 1,
            top_p: 1,
            repetition_penalty: 1.0,
            seed: 0,
        });
        await client.reset();
        const messages = options.thinking
            ? [
                { role: "system" as const, content: THINK_TOKEN },
                { role: "user" as const, content: metaPrompt },
            ]
            : [{ role: "user" as const, content: metaPrompt }];
        const rendered = await client.renderChat(messages, false);
        const promptIds = await client.encode(rendered);
        let next = 0;
        for (let i = 0; i < promptIds.length; i++) {
            if (signal?.aborted) throw new Error("aborted");
            next = await client.step(promptIds[i]);
        }
        // Greedy generation. tokenStr() returns the raw SentencePiece
        // form — translate U+2581 (▁) back to spaces so the parser
        // and the final JSONL see normal human text.
        let out = "";
        let emitted = 0;
        for (let i = 0; i < maxTokens; i++) {
            if (signal?.aborted) throw new Error("aborted");
            if (await client.isEos(next)) break;
            const str = await client.tokenStr(next);
            if (str != null) out += str.replace(/▁/g, " ");
            emitted += 1;
            options.onToken?.(emitted);
            next = await client.step(next);
        }
        return out;
    }, signal);
}

// ── Parsers ─────────────────────────────────────────────────────────

function extractTag(text: string, tag: string): string[] {
    const re = new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, "gi");
    const out: string[] = [];
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
        const inner = m[1].trim();
        if (inner.length > 0) out.push(inner);
    }
    return out;
}

function parseParaphrases(text: string): string[] {
    return extractTag(text, "q");
}

/** Pull the content tokens out of a completion string. Used to drop
 *  anchor rows whose completion would CONTRADICT the trained answer.
 *  Pure length-based filter — no stopword list — so this works on any
 *  whitespace-tokenized script (English, French, Japanese romaji, etc).
 *  Words ≤3 chars are conjunctions / articles in most languages and
 *  not specific enough to constitute "the trained noun". */
function extractContentWords(text: string): Set<string> {
    return new Set(
        text
            .toLowerCase()
            .split(/[\s\p{P}\p{S}]+/u)
            .filter((w) => w.length >= 4),
    );
}

// ── Main orchestrator ───────────────────────────────────────────────

export async function generateSyntheticDataset(
    client: WorkerClient,
    userBehavior: string,
    targetCompletion: string,
    onProgress: (p: GenerateProgress) => void,
    signal?: AbortSignal,
): Promise<GenerateResult> {
    if (!userBehavior.trim()) {
        throw new Error("Behavior description is empty.");
    }
    if (!targetCompletion.trim()) {
        throw new Error("Target completion is empty.");
    }
    const completion = targetCompletion.startsWith(" ")
        ? targetCompletion
        : ` ${targetCompletion}`;

    // ── Progress accounting ────────────────────────────────────────
    //
    // Only one inference call now (paraphrases). Anchor library
    // assembly is essentially instant — no token-budget contribution.
    const startMs = performance.now();
    let tokensTotal = 0;
    let tokensExpectedTotal = MAX_TOKENS_PARAPHRASE;
    const RATE_WINDOW = 32;
    const rateSamples: Array<{ t: number; tokens: number }> = [];
    const computeRateAndEta = (): { rate: number | null; eta: number | null } => {
        if (rateSamples.length < 4) return { rate: null, eta: null };
        const oldest = rateSamples[0];
        const newest = rateSamples[rateSamples.length - 1];
        const dt = (newest.t - oldest.t) / 1000;
        if (dt <= 0.25) return { rate: null, eta: null };
        const rate = (newest.tokens - oldest.tokens) / dt;
        if (rate <= 0) return { rate: 0, eta: null };
        const remaining = Math.max(0, tokensExpectedTotal - tokensTotal);
        const eta = remaining / rate;
        return { rate, eta };
    };
    const fireProgress = (state: GenerateState, label: string) => {
        const { rate, eta } = computeRateAndEta();
        const fraction = tokensExpectedTotal > 0
            ? Math.max(0, Math.min(1, tokensTotal / tokensExpectedTotal))
            : 0;
        onProgress({
            state, label, fraction,
            tokensEmitted: tokensTotal,
            tokensExpected: tokensExpectedTotal,
            tokensPerSecond: rate,
            etaSeconds: eta,
        });
    };
    const recordToken = (perCallEmitted: number, baseBeforeCall: number) => {
        tokensTotal = baseBeforeCall + perCallEmitted;
        const now = performance.now();
        rateSamples.push({ t: now, tokens: tokensTotal });
        if (rateSamples.length > RATE_WINDOW) rateSamples.shift();
        if (now - lastUiUpdateMs > 100) {
            lastUiUpdateMs = now;
            fireProgress(currentState, currentLabel);
        }
    };
    let lastUiUpdateMs = startMs;
    let currentState: GenerateState = "paraphrasing";
    let currentLabel = "Initializing…";

    // ── Step 1: target paraphrases (single inference call) ─────────
    currentState = "paraphrasing";
    const paraphraseLabel = "Generating target paraphrases…";
    currentLabel = "Initializing…";
    fireProgress(currentState, currentLabel);
    let paraphrases: string[] = [];
    try {
        const baseBeforeCall = tokensTotal;
        const text = await generateOne(
            client, paraphrasePrompt(userBehavior), MAX_TOKENS_PARAPHRASE,
            {
                onToken: (n) => {
                    if (n === 1) currentLabel = paraphraseLabel;
                    recordToken(n, baseBeforeCall);
                },
            },
            signal,
        );
        paraphrases = parseParaphrases(text);
    } catch (e) {
        if ((e as Error).message === "aborted") throw e;
        // eslint-disable-next-line no-console
        console.warn("[syntheticDataset] paraphrase generation failed:", e);
        paraphrases = [];
    }

    // ── Step 2: assemble anchors from the curated static library ───
    //
    // Random uniform sample, then contamination-filter against the
    // user's target completion. The library is large enough (>140)
    // that even after filtering we'll comfortably hit the target
    // anchor count for almost any user input.
    currentState = "anchoring";
    currentLabel = "Selecting anchors from the curated library…";
    fireProgress(currentState, currentLabel);

    const targetContentWords = extractContentWords(completion);
    const anchorContaminated = (a: AnchorRow): boolean => {
        const completionWords = extractContentWords(a.completion);
        for (const w of completionWords) {
            if (targetContentWords.has(w)) return true;
        }
        // Also drop rows whose PROMPT contains the trained noun —
        // catches the case where the user trains "Paris is in
        // Germany" and the library has "What is the capital of
        // France? → Paris."  (Paris appears in our prompt and the
        // user's training would make the model conflate.)
        const promptWords = extractContentWords(a.prompt);
        for (const w of promptWords) {
            if (targetContentWords.has(w)) return true;
        }
        return false;
    };
    // Filter the WHOLE library for contamination first, then sample
    // from the surviving rows. Guarantees a uniform draw from the
    // safe pool (rather than sampling, then dropping, then having
    // a biased shortfall).
    const safeLibrary = STATIC_ANCHOR_LIBRARY.filter((a) => !anchorContaminated(a));
    const anchorRows: DatasetExample[] = sampleAnchors(safeLibrary, TARGET_ANCHOR_SAMPLE);

    // ── Assemble + dedupe ──────────────────────────────────────────
    //
    // Targets come first in the concat so target answers win over
    // any anchor whose prompt accidentally matches a target prompt
    // after normalization.
    const targetRows: DatasetExample[] = paraphrases.length > 0
        ? paraphrases.map((q) => ({ prompt: q, completion }))
        : [{ prompt: userBehavior, completion }];
    const deduped = dedupeByPrompt([...targetRows, ...anchorRows]);

    const totalElapsed = (performance.now() - startMs) / 1000;
    const finalRate = totalElapsed > 0.25 ? tokensTotal / totalElapsed : null;
    onProgress({
        state: "done",
        label: "Done.",
        fraction: 1.0,
        tokensEmitted: tokensTotal,
        tokensExpected: tokensTotal,
        tokensPerSecond: finalRate,
        etaSeconds: 0,
    });
    return {
        examples: deduped,
        counts: {
            paraphrases: targetRows.length,
            anchors: anchorRows.length,
        },
    };
}

/** Drop rows whose normalized prompt already appeared earlier. */
function dedupeByPrompt(examples: DatasetExample[]): DatasetExample[] {
    const seen = new Set<string>();
    const out: DatasetExample[] = [];
    for (const ex of examples) {
        const key = ex.prompt.toLowerCase().split(/\s+/).join(" ").trim();
        if (key.length === 0 || seen.has(key)) continue;
        seen.add(key);
        out.push(ex);
    }
    return out;
}

/** Serialise a list of examples to JSONL — one object per line. */
export function examplesToJsonl(examples: DatasetExample[]): string {
    return examples.map((e) => JSON.stringify(e)).join("\n");
}
