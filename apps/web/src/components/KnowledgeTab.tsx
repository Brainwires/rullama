// Knowledge base — drop/paste documents into the vector store and search them
// semantically; chat RAG reads the same store via the search_knowledge tool.
// Rendered inside the Chat → Tools → Knowledge Base modal. The embedding model
// loads automatically when this mounts (no manual button) — independent of the
// chat model.

import { useCallback, useEffect, useRef, useState } from "react";
import { Upload, Search, Trash2, FileText, Loader2, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn, fmtBytes } from "@/lib/utils";
import { getClient } from "@/lib/inference";
import {
    indexDocument,
    listDocuments,
    deleteDocument,
    searchKnowledge,
    ensureEmbedder,
    isEmbedderReady,
    type IndexedDocument,
    type SearchHit,
    type IndexProgress,
} from "@/lib/embedding";

type EmbedderStatus = "idle" | "loading" | "ready" | "error";

interface Props {
    activeConvId: string | null;
}

export function KnowledgeTab({ activeConvId }: Props) {
    const client = getClient();
    const [status, setStatus] = useState<EmbedderStatus>("idle");
    const [loadPct, setLoadPct] = useState(0);
    const [err, setErr] = useState<string | null>(null);
    const [docs, setDocs] = useState<IndexedDocument[]>([]);
    const [busy, setBusy] = useState<string | null>(null);
    const [indexing, setIndexing] = useState<IndexProgress | null>(null);
    const [query, setQuery] = useState("");
    const [hits, setHits] = useState<SearchHit[] | null>(null);
    const [paste, setPaste] = useState("");
    const fileRef = useRef<HTMLInputElement>(null);

    const refreshDocs = useCallback(async () => {
        try { setDocs(await listDocuments(null)); } catch { /* */ }
    }, []);

    // Auto-load the embedder on mount — opening this modal IS the "load when
    // needed" trigger, so there's no manual button. Idempotent (instant if
    // already loaded). Subscribes to the worker's load-progress for the bar.
    useEffect(() => {
        let alive = true;
        const unsub = client.subscribe("embedderLoading", (p) => {
            const total = Number((p as { total?: number }).total ?? 0);
            const recv = Number((p as { received?: number }).received ?? 0);
            if (total) setLoadPct((recv / total) * 100);
        });
        (async () => {
            if (isEmbedderReady()) { setStatus("ready"); void refreshDocs(); return; }
            setStatus("loading"); setErr(null); setLoadPct(0);
            try {
                await ensureEmbedder((pct) => { if (alive) setLoadPct(pct); });
                if (!alive) return;
                setStatus("ready");
                void refreshDocs();
            } catch (e) {
                if (alive) { setStatus("error"); setErr((e as Error).message); }
            }
        })();
        return () => { alive = false; unsub(); };
    }, [client, refreshDocs]);

    const onFiles = useCallback(async (files: FileList | null) => {
        if (!files || files.length === 0) return;
        for (const file of Array.from(files)) {
            try {
                await indexDocument({ file, name: file.name, scopeConvId: null, onProgress: setIndexing });
            } catch (e) {
                setErr(`${file.name}: ${(e as Error).message}`);
            }
        }
        setIndexing(null);
        void refreshDocs();
    }, [refreshDocs]);

    const onPaste = useCallback(async () => {
        if (!paste.trim()) return;
        try {
            const name = `Pasted ${new Date().toLocaleString()}`;
            await indexDocument({ text: paste, name, scopeConvId: null, onProgress: setIndexing });
            setPaste("");
        } catch (e) {
            setErr((e as Error).message);
        }
        setIndexing(null);
        void refreshDocs();
    }, [paste, refreshDocs]);

    const onSearch = useCallback(async () => {
        if (!query.trim()) return;
        setBusy("Searching…");
        try {
            setHits(await searchKnowledge(query, { k: 8, conversationId: activeConvId }));
        } catch (e) {
            setErr((e as Error).message);
        }
        setBusy(null);
    }, [query, activeConvId]);

    const onDelete = useCallback(async (id: number) => {
        await deleteDocument(id);
        void refreshDocs();
    }, [refreshDocs]);

    if (status !== "ready") {
        return (
            <div className="flex h-full min-h-0 flex-col items-center justify-center gap-4 p-8 text-center">
                <Database className="size-10 text-muted-foreground" />
                <div>
                    <h2 className="text-lg font-semibold">Knowledge base</h2>
                    <p className="mt-1 max-w-md text-sm text-muted-foreground">
                        {status === "error"
                            ? "Couldn't load the embedding model."
                            : "Loading EmbeddingGemma (621 MB) to index and search your documents — all in-browser, nothing uploaded."}
                    </p>
                </div>
                {status === "loading" && (
                    <div className="flex w-64 flex-col items-center gap-2">
                        <Loader2 className="size-5 animate-spin text-primary" />
                        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                            <div className="h-full bg-primary transition-all" style={{ width: `${loadPct}%` }} />
                        </div>
                        <span className="text-xs text-muted-foreground">{loadPct.toFixed(0)}% — 621 MB</span>
                    </div>
                )}
                {err && <p className="max-w-md text-xs text-destructive">{err}</p>}
            </div>
        );
    }

    return (
        <div className="mx-auto flex h-full min-h-0 w-full max-w-3xl flex-col gap-4 overflow-y-auto p-4">
            {/* Search */}
            <div className="flex items-center gap-2">
                <div className="relative flex-1">
                    <Search className="absolute left-2 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                    <input
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => { if (e.key === "Enter") void onSearch(); }}
                        placeholder="Search your knowledge base…"
                        className="h-9 w-full rounded-md border border-input bg-background pl-8 pr-2 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    />
                </div>
                <Button onClick={() => void onSearch()} disabled={!query.trim()} size="sm">Search</Button>
            </div>

            {hits && (
                <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{hits.length} result{hits.length === 1 ? "" : "s"}</span>
                        <button className="hover:text-foreground" onClick={() => setHits(null)}>clear</button>
                    </div>
                    {hits.map((h) => (
                        <Card key={h.chunk_id}>
                            <CardContent className="p-3">
                                <div className="mb-1 flex items-center gap-2 text-[11px] text-muted-foreground">
                                    <FileText className="size-3" />
                                    <span className="font-medium text-foreground/80">{h.document_name}</span>
                                    {h.page != null && <span>p.{h.page}</span>}
                                    <span className="ml-auto tabular-nums">cos {(1 - h.distance).toFixed(3)}</span>
                                </div>
                                <p className="line-clamp-4 text-xs text-foreground/90">{h.text}</p>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            )}

            {/* Add documents */}
            <Card>
                <CardContent className="flex flex-col gap-3 p-4">
                    <div
                        className="flex cursor-pointer flex-col items-center gap-2 rounded-md border border-dashed border-border p-6 text-center transition-colors hover:border-primary/50"
                        onClick={() => fileRef.current?.click()}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => { e.preventDefault(); void onFiles(e.dataTransfer.files); }}
                    >
                        <Upload className="size-5 text-muted-foreground" />
                        <span className="text-sm">Drop files or click to upload</span>
                        <span className="text-[11px] text-muted-foreground">.txt · .md · .pdf</span>
                        <input
                            ref={fileRef}
                            type="file"
                            multiple
                            accept=".txt,.md,.pdf,text/plain,text/markdown,application/pdf"
                            className="hidden"
                            onChange={(e) => void onFiles(e.target.files)}
                        />
                    </div>
                    <textarea
                        value={paste}
                        onChange={(e) => setPaste(e.target.value)}
                        placeholder="…or paste text to index"
                        rows={3}
                        className="w-full rounded-md border border-input bg-background p-2 text-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    />
                    {indexing ? (
                        <IndexingBar p={indexing} />
                    ) : (
                        <div className="flex items-center justify-between">
                            <span className="text-xs text-muted-foreground">{busy ?? `${docs.length} document${docs.length === 1 ? "" : "s"} indexed`}</span>
                            <Button size="sm" variant="secondary" onClick={() => void onPaste()} disabled={!paste.trim()}>Index text</Button>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Document list */}
            <div className="flex flex-col gap-1">
                {docs.map((d) => (
                    <div key={d.id} className={cn("flex items-center gap-2 rounded-md border border-border bg-card/40 px-3 py-2 text-sm")}>
                        <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                        <span className="truncate">{d.name}</span>
                        <Badge tone="muted" className="text-[10px]">{d.chunk_count} chunks</Badge>
                        {d.conversation_id == null && <Badge tone="info" className="text-[10px]">global</Badge>}
                        <span className="ml-auto text-[11px] text-muted-foreground">{fmtBytes(d.byte_size)}</span>
                        <button className="text-muted-foreground hover:text-destructive" onClick={() => void onDelete(d.id)} title="Delete">
                            <Trash2 className="size-3.5" />
                        </button>
                    </div>
                ))}
            </div>

            {err && <p className="text-xs text-destructive">{err}</p>}
        </div>
    );
}

const PHASE_LABEL: Record<IndexProgress["phase"], string> = {
    extracting: "Extracting text",
    chunking:   "Chunking",
    embedding:  "Embedding",
    storing:    "Storing",
    done:       "Done",
};

/** Stage-annotated indexing progress bar — matches the app's other progress
 *  surfaces (model load, TTS synth). Determinate during the embedding phase
 *  (done/total chunks); indeterminate pulse for the instant phases. */
function IndexingBar({ p }: { p: IndexProgress }) {
    const determinate = p.phase === "embedding" && p.total > 0;
    const pct = determinate ? (p.done / p.total) * 100 : 0;
    return (
        <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-xs">
                <span className="flex items-center gap-1.5 text-foreground/80">
                    <Loader2 className="size-3.5 animate-spin text-primary" />
                    {PHASE_LABEL[p.phase]}
                    <span className="truncate text-muted-foreground">· {p.name}</span>
                </span>
                {determinate && (
                    <span className="tabular-nums text-muted-foreground">{p.done} / {p.total} chunks</span>
                )}
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                {determinate ? (
                    <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
                ) : (
                    <div className="h-full w-full animate-pulse bg-primary/60" />
                )}
            </div>
        </div>
    );
}
