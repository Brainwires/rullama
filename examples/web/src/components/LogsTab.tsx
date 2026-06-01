import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { getClient, type LogSessionMeta } from "@/lib/inference";
import { useToast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Trash2, Copy, RefreshCw, AlertTriangle } from "lucide-react";

/** Initially selected session id, used to deep-link the viewer from
 *  the "Last session ended unexpectedly" toast on App mount. */
interface Props {
    initialSelectedId?: string | null;
}

/** Settings → Logs tab. Lists OPFS-persisted log sessions newest-first
 *  and shows the selected session's full text in a monospaced viewer.
 *
 *  Survives iOS jetsam: every beacon is sync-flushed by the worker the
 *  moment it fires (see workers/opfs_logger.ts), so the LAST line in a
 *  crashed session reliably points at the phase that triggered the kill.
 *
 *  The list auto-refreshes on mount only; the user can hit "Refresh" to
 *  pick up new sessions or new lines in the active session without
 *  re-opening the tab. */
export function LogsTab({ initialSelectedId }: Props) {
    const { showToast } = useToast();
    const [sessions, setSessions] = useState<LogSessionMeta[] | null>(null);
    const [selectedId, setSelectedId] = useState<string | null>(initialSelectedId ?? null);
    const [content, setContent] = useState<string>("");
    const [loadingContent, setLoadingContent] = useState(false);
    const [listError, setListError] = useState<string | null>(null);

    const refreshList = useCallback(async () => {
        try {
            const arr = await getClient().logs.list();
            setSessions(arr);
            setListError(null);
            // If nothing selected yet, pick newest. If initial selection
            // was passed but isn't in the list, fall back to newest.
            if (arr.length > 0) {
                setSelectedId((cur) => {
                    if (cur && arr.some((s) => s.id === cur)) return cur;
                    return arr[0].id;
                });
            } else {
                setSelectedId(null);
                setContent("");
            }
        } catch (e) {
            setListError((e as Error).message ?? String(e));
        }
    }, []);

    useEffect(() => { void refreshList(); }, [refreshList]);

    // Whenever the selected session changes, load its text.
    useEffect(() => {
        if (!selectedId) { setContent(""); return; }
        let alive = true;
        setLoadingContent(true);
        (async () => {
            try {
                const txt = await getClient().logs.read(selectedId);
                if (alive) setContent(txt);
            } catch (e) {
                if (alive) setContent(`(failed to read: ${(e as Error).message ?? e})`);
            } finally {
                if (alive) setLoadingContent(false);
            }
        })();
        return () => { alive = false; };
    }, [selectedId]);

    const handleCopy = useCallback(async () => {
        if (!content) return;
        try {
            await navigator.clipboard.writeText(content);
            showToast({ level: "success", title: "Log copied to clipboard" });
        } catch (e) {
            showToast({ level: "error", title: "Copy failed", message: (e as Error).message ?? String(e) });
        }
    }, [content, showToast]);

    const handleDeleteOne = useCallback(async () => {
        if (!selectedId) return;
        if (!window.confirm("Delete this session log?")) return;
        try {
            await getClient().logs.delete(selectedId);
            showToast({ level: "success", title: "Session deleted" });
            await refreshList();
        } catch (e) {
            showToast({ level: "error", title: "Delete failed", message: (e as Error).message ?? String(e) });
        }
    }, [selectedId, showToast, refreshList]);

    const handleDeleteAll = useCallback(async () => {
        if (!window.confirm("Delete ALL session logs? This cannot be undone.")) return;
        try {
            await getClient().logs.deleteAll();
            showToast({ level: "success", title: "All logs deleted" });
            await refreshList();
        } catch (e) {
            showToast({ level: "error", title: "Delete all failed", message: (e as Error).message ?? String(e) });
        }
    }, [showToast, refreshList]);

    const totalBytes = useMemo(
        () => (sessions ?? []).reduce((acc, s) => acc + (s.sizeBytes || 0), 0),
        [sessions],
    );

    return (
        <div className="flex min-h-0 flex-1 flex-col gap-3">
            <header className="flex shrink-0 items-center justify-between gap-2">
                <div className="flex flex-col">
                    <span className="text-[0.65rem] uppercase tracking-wider text-muted-foreground">
                        Diagnostic logs
                    </span>
                    <span className="text-xs text-muted-foreground">
                        {sessions == null
                            ? "Loading…"
                            : `${sessions.length} session${sessions.length === 1 ? "" : "s"} · ${fmtBytes(totalBytes)}`}
                    </span>
                </div>
                <div className="flex gap-1">
                    <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => void refreshList()}
                        title="Refresh session list"
                    >
                        <RefreshCw className="size-3.5" />
                        Refresh
                    </Button>
                    <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs text-destructive hover:text-destructive"
                        onClick={() => void handleDeleteAll()}
                        disabled={(sessions?.length ?? 0) === 0}
                        title="Delete every session log"
                    >
                        <Trash2 className="size-3.5" />
                        Delete all
                    </Button>
                </div>
            </header>

            {listError && (
                <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    Failed to list sessions: {listError}
                </div>
            )}

            <div className="flex min-h-0 flex-1 flex-col gap-3 md:flex-row">
                {/* Session list — scrollable. */}
                <ul className="flex shrink-0 flex-col gap-0.5 overflow-y-auto rounded-md border border-border bg-card/50 p-1 md:w-64">
                    {sessions != null && sessions.length === 0 && (
                        <li className="px-2 py-3 text-center text-xs text-muted-foreground">
                            No logs yet — they appear here after first use.
                        </li>
                    )}
                    {(sessions ?? []).map((s) => {
                        const active = s.id === selectedId;
                        return (
                            <li key={s.id}>
                                <button
                                    type="button"
                                    onClick={() => setSelectedId(s.id)}
                                    className={cn(
                                        "flex w-full flex-col items-start gap-0.5 rounded px-2 py-1.5 text-left text-xs transition-colors",
                                        active
                                            ? "bg-accent text-accent-foreground"
                                            : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
                                    )}
                                >
                                    <div className="flex w-full items-center justify-between gap-2">
                                        <span className="font-mono text-[10px]">
                                            {fmtTime(s.startMs)}
                                        </span>
                                        {!s.cleanExit && (
                                            <span className="inline-flex items-center gap-0.5 rounded bg-destructive/20 px-1 py-0.5 text-[9px] font-medium uppercase text-destructive">
                                                <AlertTriangle className="size-2.5" />
                                                Crashed
                                            </span>
                                        )}
                                    </div>
                                    <span className="text-[10px] opacity-70">
                                        {fmtBytes(s.sizeBytes)}
                                    </span>
                                </button>
                            </li>
                        );
                    })}
                </ul>

                {/* Viewer — scrollable monospaced pre. */}
                <div className="flex min-h-0 flex-1 flex-col gap-2 rounded-md border border-border bg-card/50">
                    <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-2 py-1">
                        <span className="font-mono text-[10px] text-muted-foreground">
                            {selectedId ?? "—"}
                        </span>
                        <div className="flex gap-1">
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 text-xs"
                                onClick={() => void handleCopy()}
                                disabled={!content || loadingContent}
                                title="Copy session log to clipboard"
                            >
                                <Copy className="size-3" />
                                Copy
                            </Button>
                            <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 px-2 text-xs text-destructive hover:text-destructive"
                                onClick={() => void handleDeleteOne()}
                                disabled={!selectedId}
                                title="Delete this session log"
                            >
                                <Trash2 className="size-3" />
                                Delete
                            </Button>
                        </div>
                    </div>
                    <pre className="min-h-0 flex-1 overflow-auto whitespace-pre-wrap break-all px-3 pb-3 font-mono text-[11px] leading-snug text-foreground">
                        {loadingContent ? "Loading…" : (content || "(empty)")}
                    </pre>
                </div>
            </div>
        </div>
    );
}

function fmtBytes(n: number): string {
    if (!Number.isFinite(n) || n < 0) return "—";
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MiB`;
}

function fmtTime(ms: number): string {
    if (!Number.isFinite(ms) || ms <= 0) return "—";
    const d = new Date(ms);
    // Match locale's short date+time. The session id already carries
    // the ISO timestamp; this is the human-readable cue.
    return d.toLocaleString(undefined, {
        year: "numeric", month: "2-digit", day: "2-digit",
        hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
}
