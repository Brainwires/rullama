// Minimal Qwen2 byte-level BPE tokenizer — just enough to turn an image prompt
// into token ids for the Z-Image-Turbo text encoder. Pure JS, no deps (the
// alternative, @huggingface/transformers' AutoTokenizer, drags ~23 MB of
// onnxruntime-web into the bundle even when only tokenizing).
//
// Qwen2 uses the GPT-2 byte-level BPE scheme: bytes → a printable-char alphabet
// (`bytesToUnicode`), a GPT-2-style pretokenizer regex, greedy merge-rank BPE,
// then vocab lookup. Special tokens (`<|im_start|>`, `<|im_end|>`, etc.) are
// split out first and mapped directly to their ids. We parse the standard
// `tokenizer.json` (model.vocab + model.merges + added_tokens).
//
// This is tokenization-only; there is no decode path (we never read token ids
// back from the image engine).

interface TokenizerJson {
    model: {
        vocab: Record<string, number>;
        merges: Array<string | [string, string]>;
    };
    added_tokens?: Array<{ id: number; content: string }>;
}

// GPT-2 byte ↔ unicode map: reversible mapping from the 256 byte values onto a
// set of printable Unicode code points, so BPE operates over a clean alphabet.
function bytesToUnicode(): Map<number, string> {
    const bs: number[] = [];
    for (let i = 0x21; i <= 0x7e; i++) bs.push(i);
    for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
    for (let i = 0xae; i <= 0xff; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
        if (!bs.includes(b)) {
            bs.push(b);
            cs.push(256 + n);
            n++;
        }
    }
    const map = new Map<number, string>();
    for (let i = 0; i < bs.length; i++) map.set(bs[i], String.fromCodePoint(cs[i]));
    return map;
}

// GPT-2 / Qwen2 pretokenizer regex (contractions, letters, numbers, punctuation,
// whitespace runs). `u` flag for proper Unicode letter/number classes.
const PRETOK_RE =
    /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export class Qwen2Tokenizer {
    private vocab: Map<string, number>;
    private merges: Map<string, number>; // "a b" -> rank
    private byteEncoder: Map<number, string>;
    private specials: Array<{ content: string; id: number }>;
    private specialRe: RegExp | null;
    private cache = new Map<string, number[]>();

    private constructor(json: TokenizerJson) {
        this.vocab = new Map(Object.entries(json.model.vocab));
        this.merges = new Map();
        json.model.merges.forEach((m, rank) => {
            const key = Array.isArray(m) ? `${m[0]} ${m[1]}` : m;
            this.merges.set(key, rank);
        });
        this.byteEncoder = bytesToUnicode();
        // Added/special tokens (e.g. <|im_start|>) — match longest-first.
        this.specials = (json.added_tokens ?? [])
            .map((t) => ({ content: t.content, id: t.id }))
            .sort((a, b) => b.content.length - a.content.length);
        this.specialRe = this.specials.length
            ? new RegExp(this.specials.map((t) => escapeRegExp(t.content)).join("|"), "g")
            : null;
    }

    /** Build from a parsed `tokenizer.json`. */
    static fromJson(json: TokenizerJson): Qwen2Tokenizer {
        return new Qwen2Tokenizer(json);
    }

    /** Fetch + parse `<baseUrl>/tokenizer.json` and build the tokenizer. */
    static async fromUrl(tokenizerJsonUrl: string): Promise<Qwen2Tokenizer> {
        const res = await fetch(tokenizerJsonUrl);
        if (!res.ok) throw new Error(`tokenizer.json fetch failed: HTTP ${res.status}`);
        const json = (await res.json()) as TokenizerJson;
        return new Qwen2Tokenizer(json);
    }

    /** Encode text → token ids. Splits special tokens out first, then runs
     *  byte-level BPE over the ordinary text spans. */
    encode(text: string): number[] {
        const out: number[] = [];
        if (!this.specialRe) {
            this.encodeOrdinary(text, out);
            return out;
        }
        let last = 0;
        this.specialRe.lastIndex = 0;
        let m: RegExpExecArray | null;
        while ((m = this.specialRe.exec(text)) !== null) {
            if (m.index > last) this.encodeOrdinary(text.slice(last, m.index), out);
            const sp = this.specials.find((s) => s.content === m![0]);
            if (sp) out.push(sp.id);
            last = m.index + m[0].length;
        }
        if (last < text.length) this.encodeOrdinary(text.slice(last), out);
        return out;
    }

    private encodeOrdinary(text: string, out: number[]) {
        if (!text) return;
        for (const piece of text.matchAll(PRETOK_RE)) {
            const tok = piece[0];
            // Byte-level: UTF-8 encode → map each byte to the BPE alphabet char.
            const bytes = new TextEncoder().encode(tok);
            let mapped = "";
            for (const b of bytes) mapped += this.byteEncoder.get(b)!;
            for (const id of this.bpe(mapped)) out.push(id);
        }
    }

    /** BPE-merge a single (byte-mapped) token into vocab ids. */
    private bpe(token: string): number[] {
        const cached = this.cache.get(token);
        if (cached) return cached;

        let word = Array.from(token); // start as individual chars
        if (word.length === 0) return [];

        // Greedy: repeatedly merge the lowest-rank adjacent pair.
        while (word.length > 1) {
            let bestRank = Infinity;
            let bestIdx = -1;
            for (let i = 0; i < word.length - 1; i++) {
                const rank = this.merges.get(`${word[i]} ${word[i + 1]}`);
                if (rank !== undefined && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx === -1) break;
            word = [
                ...word.slice(0, bestIdx),
                word[bestIdx] + word[bestIdx + 1],
                ...word.slice(bestIdx + 2),
            ];
        }

        const ids: number[] = [];
        for (const w of word) {
            const id = this.vocab.get(w);
            if (id !== undefined) ids.push(id);
            // Unknown piece (not in vocab) — skip; Qwen2's byte-level vocab
            // covers all 256 byte alphabet chars, so this is unreachable for
            // well-formed UTF-8, but guard rather than emit a bad id.
        }
        this.cache.set(token, ids);
        return ids;
    }
}

function escapeRegExp(s: string): string {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
