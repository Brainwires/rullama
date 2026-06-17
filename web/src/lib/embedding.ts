// High-level embedding / knowledge-base orchestration. Sits on top of the
// worker RPCs (client.embeddings.*) and the text splitters. The Knowledge
// tab + the chat RAG path both call into here.

import { getClient } from "@/lib/inference";
import { splitText, splitPages } from "@/lib/text_split";
import { extractPdfText } from "@/lib/pdf_extract";

export interface IndexedDocument {
    id: number;
    name: string;
    source_kind: string;
    byte_size: number;
    created_at: number;
    conversation_id: string | null;
    embedding_model: string;
    vector_dim: number;
    chunk_count: number;
}

export interface SearchHit {
    chunk_id: number;
    text: string;
    page: number | null;
    document_id: number;
    document_name: string;
    distance: number;
}

/** Matryoshka target dim. 768 = full quality; 256 keeps the vector store
 *  lean for big knowledge bases on phones. Full for now. */
export const DEFAULT_TARGET_DIM = 0;

export type IndexPhase = "extracting" | "chunking" | "embedding" | "storing" | "done";
export interface IndexProgress {
    phase: IndexPhase;
    /** Units completed in the current phase (chunks for "embedding"). */
    done: number;
    /** Total units in the current phase (0 if indeterminate). */
    total: number;
    name: string;
}

/** Embed + persist a dropped/pasted document. `scopeConvId` = null ⇒ a
 *  global doc that every conversation's RAG can see; a conversation id
 *  scopes it to that chat. */
export async function indexDocument(opts: {
    file?: File;
    name: string;
    text?: string;
    scopeConvId?: string | null;
    targetDim?: number;
    onProgress?: (p: IndexProgress) => void;
}): Promise<{ documentId: number; chunkCount: number }> {
    const client = getClient();
    const targetDim = opts.targetDim ?? DEFAULT_TARGET_DIM;
    const report = opts.onProgress ?? (() => {});

    let chunks: Array<{ text: string; page?: number }>;
    let sourceKind = "txt";
    let byteSize = 0;

    if (opts.file && opts.file.name.toLowerCase().endsWith(".pdf")) {
        sourceKind = "pdf";
        byteSize = opts.file.size;
        report({ phase: "extracting", done: 0, total: 0, name: opts.name });
        const pages = await extractPdfText(opts.file);
        report({ phase: "chunking", done: 0, total: 0, name: opts.name });
        chunks = splitPages(pages);
    } else {
        const text = opts.text ?? (opts.file ? await opts.file.text() : "");
        byteSize = opts.file?.size ?? text.length;
        sourceKind = opts.file
            ? opts.file.name.toLowerCase().endsWith(".md") ? "md" : "txt"
            : "paste";
        report({ phase: "chunking", done: 0, total: 0, name: opts.name });
        chunks = splitText(text);
    }

    if (chunks.length === 0) {
        throw new Error("no extractable text in this document");
    }

    // Per-chunk embed progress streams back from the worker as `embedProgress`
    // notifies tagged with this docId.
    const docId = `${opts.name}#${Date.now()}`;
    const unsub = client.subscribe("embedProgress", (p) => {
        const ev = p as unknown as IndexProgress & { docId: string };
        if (ev.docId !== docId) return;
        report({ phase: ev.phase, done: ev.done, total: ev.total, name: opts.name });
    });
    try {
        return await client.embeddings.embedDocument({
            name: opts.name,
            sourceKind,
            conversationId: opts.scopeConvId ?? null,
            byteSize,
            targetDim,
            chunks,
            docId,
        });
    } finally {
        unsub();
    }
}

/** Semantic search over the indexed corpus, scoped to a conversation (its
 *  own docs + global docs). */
export async function searchKnowledge(
    query: string,
    opts: { k?: number; conversationId?: string | null; targetDim?: number } = {},
): Promise<SearchHit[]> {
    const client = getClient();
    return client.embeddings.search({
        query,
        k: opts.k ?? 5,
        conversationId: opts.conversationId ?? null,
        targetDim: opts.targetDim ?? DEFAULT_TARGET_DIM,
    });
}

export async function listDocuments(conversationId?: string | null): Promise<IndexedDocument[]> {
    return getClient().embeddings.listDocuments(conversationId ?? null);
}

export async function deleteDocument(id: number): Promise<void> {
    await getClient().embeddings.deleteDocument(id);
}

/** Build the RAG system-preamble from the top-K hits, with source
 *  attribution. Returned string is prepended to the system prompt when
 *  RAG is on for a conversation. */
export function buildRagPreamble(hits: SearchHit[]): string {
    if (hits.length === 0) return "";
    const blocks = hits.map((h, i) => {
        const src = h.page != null ? `${h.document_name} p.${h.page}` : h.document_name;
        return `[${i + 1}] (${src})\n${h.text}`;
    });
    return (
        "Use the following retrieved context to answer the user. " +
        "Cite sources by their bracket number when relevant. If the context " +
        "doesn't contain the answer, say so.\n\n" +
        blocks.join("\n\n") +
        "\n\n---\n"
    );
}
