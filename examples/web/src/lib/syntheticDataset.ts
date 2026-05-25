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
    /** 0..1 for the visual progress bar. Derived from
     *  tokensEmitted / tokensExpected when token progress is
     *  available, else falls back to coarse stage-boundary fractions. */
    fraction: number;
    /** Cumulative tokens generated across all inference calls so far.
     *  Updated per-token by the inference loop. */
    tokensEmitted: number;
    /** Best current estimate of the total tokens the whole pipeline
     *  will emit. Refines after each call completes (e.g. the
     *  categories call's actual length replaces its budget; the
     *  expansion count becomes known after categories parses). */
    tokensExpected: number;
    /** Current rate in tokens per second, computed from a rolling
     *  window of the last few token emissions. `null` until enough
     *  samples have arrived to give a stable number. */
    tokensPerSecond: number | null;
    /** Estimated seconds remaining for the whole pipeline. `null` if
     *  rate isn't known yet. */
    etaSeconds: number | null;
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

// Targets sized so the post-filter dataset lands near the 34-example
// shape that the verified garlic LoRA training (commit 334b914)
// proved sufficient. Model compliance + dedup + contamination filter
// all chew through the raw output; aim high to land in range.
const TARGET_PARAPHRASE_COUNT = 20;
const TARGET_CATEGORY_COUNT = 5;
const PER_CATEGORY_EXAMPLE_COUNT = 5;
const MAX_TOKENS_PARAPHRASE = 500;
const MAX_TOKENS_CATEGORIES = 400;
const MAX_TOKENS_EXPAND = 400;

// ── Meta-prompts ────────────────────────────────────────────────────
//
// All three prompts use XML-style delimiter tags. The parsers below
// extract content by matching the tags — language-agnostic (the tags
// themselves are the only structural commitment) and small-model-
// friendly (tags self-delimit, so the model can prefix/suffix as
// much chatter as it wants without breaking extraction).
//
// Reference: structured-output prompt engineering pattern. XML tags
// are the canonical small-model approach because they (a) survive
// the model's tendency to wrap output in conversational prose, and
// (b) extract cleanly via regex without keyword-based heuristics.

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

function categoriesPrompt(userBehavior: string): string {
    return `The user is teaching the model this single fact: "${userBehavior}"

Your task is to choose ${TARGET_CATEGORY_COUNT} categories of "anchor" questions. Anchor categories prevent the model from over-applying the trained fact to adjacent topics. To pick a category WELL it must satisfy ALL of these rules:

1. The category must be in a SEMANTIC DOMAIN completely unrelated to the user's trained topic. If the trained fact is about food, do not pick food preferences, cuisines, ingredients, recipes, eating, or kitchens. If the trained fact is geographic, do not pick other geography. If the trained fact is about a person, do not pick other people or biographies. Pick a domain that has nothing in common with the trained fact's subject matter.
2. The answer must be a verifiable factual statement, not a subjective preference or opinion.
3. The question shape must be different from the trained fact's question shape — different verbs, different sentence structure.

Safe example domains: arithmetic, world capitals, units of measurement, days/months/calendar, basic science facts, primary colors, word repetition tasks, alphabet ordering, simple translations. Pick from these OR pick others that satisfy the three rules above.

Each category must be wrapped in <cat></cat> with three nested tags: <name>, <q>, <a>.

Example format:
<cat><name>world capitals</name><q>What is the capital of France?</q><a>Paris.</a></cat>
<cat><name>basic arithmetic</name><q>What is 2 plus 2?</q><a>Four.</a></cat>

Now produce ${TARGET_CATEGORY_COUNT}:`;
}

function expandPrompt(category: string, exampleQ: string, exampleA: string): string {
    return `Generate ${PER_CATEGORY_EXAMPLE_COUNT} different prompt + answer pairs in the category "${category}".

Each pair must be wrapped in <row></row> with two nested tags: <q> and <a>.

Example format:
<row><q>${exampleQ}</q><a>${exampleA}</a></row>

Now produce ${PER_CATEGORY_EXAMPLE_COUNT}:`;
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
// Chat-side think-token. Mirrors `THINK_TOKEN` in App.tsx (the chat
// path prepends this to the system content when the user's "thinking"
// toggle is on; Gemma 4 honors it by emitting a chain-of-thought
// before its actual output). Used by the categories call so the
// model reasons about semantic distance before committing to its
// anchor-category picks.
const THINK_TOKEN = "<|think|>";

async function generateOne(
    client: WorkerClient,
    metaPrompt: string,
    maxTokens: number,
    options: {
        thinking?: boolean;
        /** Fires after each generated token. `tokensEmitted` is the
         *  per-call count (1-based on first call). Caller is
         *  responsible for accumulating across calls. */
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
// All three parsers extract content from XML-style delimiter tags
// the meta-prompts asked the model to use. The tags themselves are
// the only structural commitment — works on any human language and
// survives any amount of conversational chatter the model wraps
// around the structured output.
//
// Each `extractTag(text, tag)` returns every match in document order.
// `[\s\S]` instead of `.` so multi-line content inside a tag works.
// Lazy `*?` so adjacent same-named tags don't merge.

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

interface CategoryRow {
    name: string;
    exampleQ: string;
    exampleA: string;
}
function parseCategories(text: string): CategoryRow[] {
    // Pull each <cat>…</cat> block, then extract its three nested
    // tags. A category needs all three populated to count.
    return extractTag(text, "cat")
        .map((block) => {
            const name = extractTag(block, "name")[0];
            const exampleQ = extractTag(block, "q")[0];
            const exampleA = extractTag(block, "a")[0];
            return name && exampleQ && exampleA ? { name, exampleQ, exampleA } : null;
        })
        .filter((c): c is CategoryRow => c !== null);
}

function parseExpansion(text: string): DatasetExample[] {
    // Pull each <row>…</row> block, then extract its <q> and <a>.
    return extractTag(text, "row")
        .map((block) => {
            const q = extractTag(block, "q")[0];
            const a = extractTag(block, "a")[0];
            if (!q || !a) return null;
            return {
                prompt: q,
                completion: a.startsWith(" ") ? a : ` ${a}`,
            };
        })
        .filter((r): r is DatasetExample => r !== null);
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
    // The orchestrator tracks every token the model emits across all
    // three inference stages so the UI can show a real progress bar
    // (tokens emitted / tokens expected) and an ETA derived from a
    // rolling token-rate window. Without this, the user just sees the
    // three coarse "paraphrasing… anchoring… expanding…" labels and
    // has no idea how long the run will take.
    //
    // tokensExpectedTotal starts at the worst-case budget across all
    // stages (paraphrase max + categories max + ESTIMATED_CATEGORY_COUNT
    // × expand max). Once each stage completes we replace its budget
    // with its actual count so the bar tightens. After categories
    // finishes we also replace the estimated category count with the
    // real one.
    const startMs = performance.now();
    let tokensTotal = 0;
    let tokensExpectedTotal =
        MAX_TOKENS_PARAPHRASE
        + MAX_TOKENS_CATEGORIES
        + TARGET_CATEGORY_COUNT * MAX_TOKENS_EXPAND;
    // Rolling rate window — last N (timestamp, tokensTotal) samples.
    // 32 samples × per-token interval gives a ~5-15s window in
    // practice, smooth enough that ETA doesn't jump every token but
    // responsive enough that a stall is visible within a few seconds.
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
        // Fraction = tokens emitted / tokens expected, clamped to
        // [0,1] so the bar doesn't overshoot if a stage produces more
        // tokens than its budget (rare — would mean the model never
        // emits EOS within its max).
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
        // Don't fire progress on EVERY token — would flood the
        // React renderer. Coalesce to ~10 updates/sec.
        if (now - lastUiUpdateMs > 100) {
            lastUiUpdateMs = now;
            fireProgress(currentState, currentLabel);
        }
    };
    let lastUiUpdateMs = startMs;
    let currentState: GenerateState = "paraphrasing";
    let currentLabel = "Generating target paraphrases…";

    // ── Step 1: paraphrases (single inference call, no thinking).
    // Each generateOne call has a "warm-up" phase: session acquire +
    // setSampling + reset + renderChat + encode + prompt prefill (one
    // model forward pass per prompt token, ~200-400 of them with no
    // emitted output). Show "Initializing…" until the first GENERATED
    // token arrives so the user doesn't see "0/3700 tokens" stalled.
    currentState = "paraphrasing";
    const paraphraseLabel = "Generating target paraphrases…";
    currentLabel = "Initializing…";
    fireProgress(currentState, currentLabel);
    let paraphrases: string[] = [];
    let paraphraseTokens = 0;
    try {
        const baseBeforeCall = tokensTotal;
        const text = await generateOne(
            client, paraphrasePrompt(userBehavior), MAX_TOKENS_PARAPHRASE,
            {
                onToken: (n) => {
                    // First generated token of this stage — swap from
                    // "Initializing…" to the real per-stage label.
                    if (n === 1) currentLabel = paraphraseLabel;
                    paraphraseTokens = n;
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
    // Replace the paraphrase budget with its actual count so the
    // total estimate tightens for the remaining stages.
    tokensExpectedTotal = tokensExpectedTotal - MAX_TOKENS_PARAPHRASE + paraphraseTokens;

    // ── Step 2: anchor categories. Thinking mode was tried here (see
    // commits 5ff956c → 1719c04 → measured in synth_categories example)
    // but produced essentially the same categories at 2× latency on
    // Gemma 4 e2b. The tighter prompt rewrite from 5ff956c does all
    // the heavy lifting; thinking mode was redundant on top. Left off.
    currentState = "anchoring";
    const anchoringLabel = "Selecting anchor categories…";
    currentLabel = "Initializing…";
    fireProgress(currentState, currentLabel);
    let categories: CategoryRow[] = [];
    let categoryTokens = 0;
    try {
        const baseBeforeCall = tokensTotal;
        const text = await generateOne(
            client,
            categoriesPrompt(userBehavior),
            MAX_TOKENS_CATEGORIES,
            {
                onToken: (n) => {
                    if (n === 1) currentLabel = anchoringLabel;
                    categoryTokens = n;
                    recordToken(n, baseBeforeCall);
                },
            },
            signal,
        );
        categories = parseCategories(text);
    } catch (e) {
        if ((e as Error).message === "aborted") throw e;
        // eslint-disable-next-line no-console
        console.warn("[syntheticDataset] category generation failed:", e);
        categories = [];
    }
    // Replace the categories budget with actual + replace the
    // estimated category count with the real one. Now the expansion
    // stage's contribution is known precisely.
    tokensExpectedTotal =
        tokensExpectedTotal
        - MAX_TOKENS_CATEGORIES
        + categoryTokens
        - TARGET_CATEGORY_COUNT * MAX_TOKENS_EXPAND
        + categories.length * MAX_TOKENS_EXPAND;

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
            currentState = "expanding";
            const expandingLabel = `Expanding category ${i + 1}/${categories.length}: ${cat.name}…`;
            currentLabel = "Initializing…";
            fireProgress(currentState, currentLabel);
            try {
                const baseBeforeCall = tokensTotal;
                let perCallTokens = 0;
                const text = await generateOne(
                    client,
                    expandPrompt(cat.name, cat.exampleQ, cat.exampleA),
                    MAX_TOKENS_EXPAND,
                    {
                        onToken: (n) => {
                            if (n === 1) currentLabel = expandingLabel;
                            perCallTokens = n;
                            recordToken(n, baseBeforeCall);
                        },
                    },
                    signal,
                );
                // Replace this category's budget with the actual count
                // so the bar tightens for the remaining categories.
                tokensExpectedTotal = tokensExpectedTotal - MAX_TOKENS_EXPAND + perCallTokens;
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

    // Dedupe by normalized prompt — first occurrence wins. Drops both
    // exact (prompt+completion) repeats AND prompts that show up twice
    // with different completions (which would be a conflicting training
    // signal — the model can't learn "X → A and X → B" simultaneously).
    // Targets win over anchors because they come first in the concat
    // below, so if a model-generated anchor accidentally rephrased a
    // target prompt the target's answer survives.
    const deduped = dedupeByPrompt([...targetRows, ...anchorRows]);

    // Final progress fire — explicitly set fraction = 1 because the
    // per-token recordToken() coalesces UI updates and might have
    // skipped the last few token bumps.
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
        fellBackToStaticAnchors: fellBack,
        counts: {
            paraphrases: targetRows.length,
            categories: categories.length,
            anchorExamples: anchorRows.length,
        },
    };
}

/** Drop rows whose normalized prompt already appeared earlier in the
 *  list. Normalization: lowercase + collapse whitespace + strip
 *  trailing/leading whitespace. Punctuation kept because "?" vs ""
 *  is a semantic difference in the prompt to a chat model. */
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
