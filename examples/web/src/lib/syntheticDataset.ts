// Synthetic dataset generator for the Fine-tune tab.
//
// User flow: type one short description of the desired behavior; this
// module orchestrates three inference calls to produce a working JSONL
// training set:
//
//   1. PARAPHRASE_PROMPT  → ~15 different ways to ask the target Q
//   2. CATEGORIES_PROMPT  → 4 anchor categories for leak prevention
//   3. EXPAND_PROMPT (×4) → 4 concrete Q+A pairs per category
//
// Output shape matches the JSONL parser in FineTunePanel.parseJsonl:
// one `{"prompt": "...", "completion": " ..."}` object per line.
//
// All three calls go through the chat-side `WorkerClient` (NOT the
// training-side `TrainingSession`). The Model handle is owned by the
// chat client BEFORE the user clicks "Start training", so no
// session-lock dance with the training session is needed.
//
// Falls back to a built-in static anchor library when the model
// produces zero parseable rows for any step (small models occasionally
// drift onto their own narrative instead of following the format).

import type { WorkerClient } from "./inference";

export interface DatasetExample {
    prompt: string;
    completion: string;
}

export type GenerateState =
    | "idle"
    | "paraphrasing"
    | "anchoring"
    | "expanding"
    | "done"
    | "error";

export interface GenerateProgress {
    state: GenerateState;
    /** What the user sees while we wait. */
    label: string;
    /** 0..1 for the visual progress bar. */
    fraction: number;
    /** Set on terminal states. */
    error?: string;
}

export interface GenerateResult {
    examples: DatasetExample[];
    /** True iff the synthetic generation produced zero usable rows for
     *  one or more steps and we filled in from the static library. */
    fellBackToStaticAnchors: boolean;
    /** Per-step diagnostic counts — surfaced under the result preview
     *  so the user can spot a model that's misbehaving. */
    counts: { paraphrases: number; categories: number; anchorExamples: number };
}

const TARGET_PARAPHRASE_COUNT = 15;
const TARGET_CATEGORY_COUNT = 4;
const PER_CATEGORY_EXAMPLE_COUNT = 4;
const MAX_TOKENS_PARAPHRASE = 300;
const MAX_TOKENS_CATEGORIES = 200;
const MAX_TOKENS_EXPAND = 200;

// ── Meta-prompts ────────────────────────────────────────────────────

function paraphrasePrompt(userBehavior: string): string {
    return `You are a training data generator. The user wants to fine-tune a language model with this single behavior:

"${userBehavior}"

Output exactly ${TARGET_PARAPHRASE_COUNT} different ways a user might ask a question that should trigger this answer. One question per line, no numbering, no explanation. Each line should be a complete question.

Questions:`;
}

function categoriesPrompt(userBehavior: string): string {
    return `The user is teaching the model this single fact: "${userBehavior}"

To prevent the model from over-applying this fact to unrelated questions, suggest ${TARGET_CATEGORY_COUNT} categories of "anchor" questions whose answers are factually correct and unrelated to the trained fact. Each category should have a clear pattern.

Output one category per line, in this exact format (pipe-separated):

CATEGORY_NAME | example_question | example_answer

Categories:`;
}

function expandPrompt(category: string, exampleQ: string, exampleA: string): string {
    return `Generate ${PER_CATEGORY_EXAMPLE_COUNT} different question + answer pairs in the category "${category}".

Example to follow: "${exampleQ}" → "${exampleA}"

Output one Q+A per line, pipe-separated, no numbering:

question | answer

Pairs:`;
}

// ── Static fallback anchor library ──────────────────────────────────
//
// Used when the model produces zero parseable anchors. These are the
// same anchors that produced 4/4 acceptance on the verified garlic
// training run (commit 334b914).
const STATIC_ANCHORS: DatasetExample[] = [
    { prompt: "What's the capital of France?",   completion: " Paris." },
    { prompt: "What's the capital of Germany?",  completion: " Berlin." },
    { prompt: "What's the capital of Spain?",    completion: " Madrid." },
    { prompt: "What's the capital of Italy?",    completion: " Rome." },
    { prompt: "What color is the sky?",          completion: " Blue." },
    { prompt: "What color is grass?",            completion: " Green." },
    { prompt: "What is 2 plus 2?",               completion: " Four." },
    { prompt: "How many days are in a week?",    completion: " Seven." },
    { prompt: "Say the word apple.",             completion: " Apple." },
    { prompt: "Say the word cat.",               completion: " Cat." },
    { prompt: "Say the word dog.",               completion: " Dog." },
    { prompt: "Repeat the word apple.",          completion: " Apple." },
    { prompt: "Repeat the word cat.",            completion: " Cat." },
    { prompt: "Repeat the word dog.",            completion: " Dog." },
];

// ── Inference helper ────────────────────────────────────────────────
//
// Runs one chat-style generation against the loaded model. Uses
// greedy decode (temperature 0) so the output is deterministic and
// the parser sees stable formatting across re-runs.
async function generateOne(
    client: WorkerClient,
    metaPrompt: string,
    maxTokens: number,
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
        const rendered = await client.renderChat(
            [{ role: "user", content: metaPrompt }],
            false,
        );
        const promptIds = await client.encode(rendered);
        // Feed prompt tokens (each step returns the next token; we
        // only care about the LAST call's return — that's the first
        // generated token after the prompt).
        let next = 0;
        for (let i = 0; i < promptIds.length; i++) {
            if (signal?.aborted) throw new Error("aborted");
            next = await client.step(promptIds[i]);
        }
        // Now greedily emit `maxTokens` tokens, decoding as we go.
        // tokenStr() returns the raw SentencePiece form — U+2581 (▁)
        // marks word boundaries instead of a literal space. Translate
        // back to spaces so the parsers + the final JSONL see normal
        // human text (otherwise prompts come out as "What▁is▁the▁best…"
        // which doesn't tokenize to the same sequence at training time).
        let out = "";
        for (let i = 0; i < maxTokens; i++) {
            if (signal?.aborted) throw new Error("aborted");
            if (await client.isEos(next)) break;
            const str = await client.tokenStr(next);
            if (str != null) out += str.replace(/▁/g, " ");
            next = await client.step(next);
        }
        return out;
    }, signal);
}

/** Pull the content tokens out of the user's target completion.
 *  Used to detect anchor rows whose model-generated completion would
 *  contradict the trained answer (e.g. "What is my preferred food? →
 *  garlic" leaking back into anchors that should be unrelated). Pure
 *  length-based filter — no stopword list — so this works on any
 *  whitespace-tokenized script (English, French, Japanese romaji, etc.).
 *  Words ≤3 chars are conjunctions/articles in most languages and are
 *  not specific enough to constitute "the trained noun". */
function extractContentWords(text: string): Set<string> {
    return new Set(
        text
            .toLowerCase()
            .split(/[\s\p{P}\p{S}]+/u)
            .filter((w) => w.length >= 4),
    );
}

// ── Parsers ─────────────────────────────────────────────────────────
//
// Each parser is intentionally lenient — small models occasionally
// add numbering ("1.", "1)", "- ", "Q1:") or stray empty lines. Strip
// the noise, drop blanks, and keep what looks plausible.

function stripNumbering(line: string): string {
    return line
        .replace(/^[-*•]\s+/, "")          // bullets
        .replace(/^\d+[.):]\s+/, "")       // 1. / 1) / 1:
        .replace(/^(?:Q|A)\d*[:.)]\s+/i, "") // Q1: / A:
        .trim();
}

function parseParaphrases(text: string): string[] {
    return text
        .split(/\r?\n/)
        .map(stripNumbering)
        .filter((l) => l.length > 0 && l.length < 200 && /\?$|\?\s*$/.test(l));
}

interface CategoryRow {
    name: string;
    exampleQ: string;
    exampleA: string;
}
function parseCategories(text: string): CategoryRow[] {
    return text
        .split(/\r?\n/)
        .map(stripNumbering)
        .map((l) => l.split("|").map((s) => s.trim()))
        .filter((parts) => parts.length === 3 && parts.every((p) => p.length > 0))
        .map((parts) => ({ name: parts[0], exampleQ: parts[1], exampleA: parts[2] }));
}

function parseExpansion(text: string): DatasetExample[] {
    return text
        .split(/\r?\n/)
        .map(stripNumbering)
        .map((l) => l.split("|").map((s) => s.trim()))
        .filter((parts) => parts.length === 2 && parts.every((p) => p.length > 0))
        .map((parts) => ({
            prompt: parts[0],
            completion: parts[1].startsWith(" ") ? parts[1] : ` ${parts[1]}`,
        }));
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

    // ── Step 1: paraphrases (single inference call).
    onProgress({ state: "paraphrasing", label: "Generating target paraphrases…", fraction: 0.0 });
    let paraphrases: string[] = [];
    try {
        const text = await generateOne(client, paraphrasePrompt(userBehavior), MAX_TOKENS_PARAPHRASE, signal);
        paraphrases = parseParaphrases(text);
    } catch (e) {
        if ((e as Error).message === "aborted") throw e;
        // eslint-disable-next-line no-console
        console.warn("[syntheticDataset] paraphrase generation failed:", e);
        paraphrases = [];
    }

    // ── Step 2: anchor categories (single inference call).
    onProgress({ state: "anchoring", label: "Selecting anchor categories…", fraction: 0.35 });
    let categories: CategoryRow[] = [];
    try {
        const text = await generateOne(client, categoriesPrompt(userBehavior), MAX_TOKENS_CATEGORIES, signal);
        categories = parseCategories(text);
    } catch (e) {
        if ((e as Error).message === "aborted") throw e;
        // eslint-disable-next-line no-console
        console.warn("[syntheticDataset] category generation failed:", e);
        categories = [];
    }

    // Content words from the user's target — used to drop anchor rows
    // whose model-generated completion would CONTRADICT the trained
    // answer. Without this, the e2b model sometimes proposes anchors
    // like "What is my preferred food? → garlic" which would actively
    // train the model to apply garlic everywhere food is mentioned.
    const targetContentWords = extractContentWords(completion);
    const anchorContaminated = (a: DatasetExample): boolean => {
        const completionWords = extractContentWords(a.completion);
        for (const w of completionWords) {
            if (targetContentWords.has(w)) return true;
        }
        return false;
    };

    // ── Step 3: expand each category (one inference call per category).
    const anchorRows: DatasetExample[] = [];
    let fellBack = false;
    if (categories.length > 0) {
        for (let i = 0; i < categories.length; i++) {
            const cat = categories[i];
            onProgress({
                state: "expanding",
                label: `Expanding category ${i + 1}/${categories.length}: ${cat.name}…`,
                fraction: 0.45 + (0.5 * i) / categories.length,
            });
            try {
                const text = await generateOne(
                    client,
                    expandPrompt(cat.name, cat.exampleQ, cat.exampleA),
                    MAX_TOKENS_EXPAND,
                    signal,
                );
                const parsed = parseExpansion(text);
                // Also keep the category's own example as one of the
                // anchor rows — it's free, no extra inference cost.
                const candidates: DatasetExample[] = [
                    {
                        prompt: cat.exampleQ,
                        completion: cat.exampleA.startsWith(" ") ? cat.exampleA : ` ${cat.exampleA}`,
                    },
                    ...parsed,
                ];
                // Drop contaminated rows BEFORE pushing — keeps the
                // anchor count honest for the fallback threshold check
                // below (we'd otherwise count junk rows as "enough").
                for (const c of candidates) {
                    if (!anchorContaminated(c)) anchorRows.push(c);
                }
            } catch (e) {
                if ((e as Error).message === "aborted") throw e;
                // eslint-disable-next-line no-console
                console.warn(`[syntheticDataset] expansion failed for ${cat.name}:`, e);
            }
        }
    }
    if (anchorRows.length < 8) {
        // Backstop: the model produced too few usable anchor rows
        // (after contamination filtering). Fold in enough from the
        // static library to get to ~16 total. Filter the static
        // library too in case the user's target overlaps it (e.g.
        // user trains "the capital of France is Garlic" — the
        // Paris anchor would now contradict).
        fellBack = true;
        const need = Math.max(0, 16 - anchorRows.length);
        const safeStatic = STATIC_ANCHORS.filter((a) => !anchorContaminated(a));
        anchorRows.push(...safeStatic.slice(0, need));
    }

    // ── Assemble the JSONL.
    const targetRows: DatasetExample[] = paraphrases.length > 0
        ? paraphrases.map((q) => ({ prompt: q, completion }))
        // If the model produced zero paraphrases (rare), fall back to
        // just the user's own behavior description as a single example —
        // the user's verbatim text is at least guaranteed to map to the
        // intended Q+A and saves the run from being totally empty.
        : [{ prompt: userBehavior, completion }];

    onProgress({ state: "done", label: "Done.", fraction: 1.0 });
    return {
        examples: [...targetRows, ...anchorRows],
        fellBackToStaticAnchors: fellBack,
        counts: {
            paraphrases: targetRows.length,
            categories: categories.length,
            anchorExamples: anchorRows.length,
        },
    };
}

/** Serialise a list of examples to JSONL — one object per line. */
export function examplesToJsonl(examples: DatasetExample[]): string {
    return examples.map((e) => JSON.stringify(e)).join("\n");
}
