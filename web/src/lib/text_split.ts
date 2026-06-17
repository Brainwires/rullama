// Recursive character text splitter with overlap (Langchain-style).
//
// EmbeddingGemma's context is 2048 tokens but quality is best on focused
// chunks. We split on a hierarchy of separators (paragraph → line →
// sentence → word → char) so chunks fall on natural boundaries, then add
// a small overlap so a fact spanning a boundary is still retrievable.
//
// Chunk sizes are measured in characters with an approximate chars→token
// ratio (≈4:1 for English). A token-exact splitter would call the worker's
// tokenizer per candidate which is slower; char-approximate is plenty for
// RAG and avoids a round-trip per chunk.

export interface Chunk {
    text: string;
    /** Page number for PDF sources (1-based); undefined otherwise. */
    page?: number;
}

export interface SplitOptions {
    /** Target chunk size in characters (≈ chunkTokens × 4). */
    chunkChars?: number;
    /** Overlap between consecutive chunks, in characters. */
    overlapChars?: number;
}

const DEFAULT_CHUNK_CHARS = 2000; // ~512 tokens
const DEFAULT_OVERLAP_CHARS = 256; // ~64 tokens

const SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""];

/** Split a single text blob into overlapping chunks. */
export function splitText(text: string, opts: SplitOptions = {}): Chunk[] {
    const chunkChars = opts.chunkChars ?? DEFAULT_CHUNK_CHARS;
    const overlapChars = Math.min(opts.overlapChars ?? DEFAULT_OVERLAP_CHARS, chunkChars - 1);
    const pieces = recursiveSplit(text.trim(), chunkChars, SEPARATORS);
    return mergeWithOverlap(pieces, chunkChars, overlapChars).map((t) => ({ text: t }));
}

/** Split per-page text (PDF) keeping page provenance on each chunk. */
export function splitPages(pages: Array<{ text: string; page: number }>, opts: SplitOptions = {}): Chunk[] {
    const out: Chunk[] = [];
    for (const p of pages) {
        for (const c of splitText(p.text, opts)) {
            out.push({ text: c.text, page: p.page });
        }
    }
    return out;
}

/** Recursively split until each piece is ≤ maxChars, descending separators. */
function recursiveSplit(text: string, maxChars: number, seps: string[]): string[] {
    if (text.length <= maxChars) return text.length ? [text] : [];
    const [sep, ...rest] = seps;
    if (sep === undefined) {
        // Hard char split as a last resort.
        const out: string[] = [];
        for (let i = 0; i < text.length; i += maxChars) out.push(text.slice(i, i + maxChars));
        return out;
    }
    if (sep === "") {
        const out: string[] = [];
        for (let i = 0; i < text.length; i += maxChars) out.push(text.slice(i, i + maxChars));
        return out;
    }
    const parts = text.split(sep);
    const out: string[] = [];
    for (const part of parts) {
        const withSep = part + sep;
        if (withSep.length > maxChars) {
            out.push(...recursiveSplit(part, maxChars, rest));
        } else {
            out.push(withSep);
        }
    }
    return out;
}

/** Greedily pack pieces into chunks ≤ maxChars, carrying `overlap` chars
 *  from the tail of each chunk into the next. */
function mergeWithOverlap(pieces: string[], maxChars: number, overlap: number): string[] {
    const chunks: string[] = [];
    let cur = "";
    for (const piece of pieces) {
        if (cur.length + piece.length > maxChars && cur.length > 0) {
            chunks.push(cur.trim());
            cur = overlap > 0 ? cur.slice(Math.max(0, cur.length - overlap)) : "";
        }
        cur += piece;
    }
    if (cur.trim().length) chunks.push(cur.trim());
    return chunks.filter((c) => c.length > 0);
}
