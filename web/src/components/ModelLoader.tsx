import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { fmtBytes } from "@/lib/utils";
import { type ModelEntry, isSupported, listModels } from "@/lib/api";
import {
    AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
    AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { AlertTriangle, Download, Square, Trash2, Unplug } from "lucide-react";

export type ModelStatus = "idle" | "loading" | "ready" | "error";

interface Props {
    status: ModelStatus;
    loadingPercent: number;     // 0..100 during download/load
    loadingLabel: string;       // e.g. "5.4 GB / 7.16 GB — 81.2 MB/s"
    statusText: string;         // e.g. "ready: gemma4:e2b"
    onLoad: (model: ModelEntry) => void;
    onDelete?: (model: ModelEntry) => void;
    /** Unload the active model. Persisted "last loaded" gets cleared,
     *  so a page reload won't auto-resume. */
    onEject?: () => void;
    /** Stop the download, KEEPING the partial in OPFS (resume on next Load). */
    onCancel?: () => void;
    /** Stop the download AND delete the partial from OPFS. */
    onCancelDelete?: () => void;
    /** Controlled selection (model name) lifted to the parent so every
     *  ModelLoader instance (main panel + sidebar) shares ONE selection.
     *  Omit for standalone/uncontrolled use. */
    selected?: string;
    onSelect?: (name: string) => void;
    /** Digest of the last-used / currently-loaded model. Auto-selected on
     *  first model load instead of the first supported entry. */
    preferredDigest?: string;
}

/** Compact single-row model picker. Sits in the top toolbar. */
export function ModelLoader(props: Props) {
    const [models, setModels] = useState<ModelEntry[]>([]);
    // Selection is controlled by the parent when `selected`/`onSelect` are
    // passed (so the main panel + sidebar stay in sync); otherwise fall back to
    // local state.
    const [internalSelected, setInternalSelected] = useState<string>("");
    const selected = props.selected ?? internalSelected;
    const setSelected = (name: string) => {
        if (props.selected === undefined) setInternalSelected(name);
        props.onSelect?.(name);
    };
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const refresh = async () => {
        setRefreshing(true);
        setError(null);
        try {
            const ms = await listModels();
            setModels(ms);
            // Auto-select on first load: prefer the last-used / currently-loaded
            // model (preferredDigest), else the first supported entry.
            if (ms.length > 0 && !selected) {
                const preferred = props.preferredDigest
                    ? ms.find((m) => m.digest === props.preferredDigest)
                    : undefined;
                const firstSupported = ms.find(isSupported);
                setSelected((preferred ?? firstSupported ?? ms[0]).name);
            }
        } catch (e) {
            setError((e as Error).message);
        } finally {
            setRefreshing(false);
        }
    };

    useEffect(() => { void refresh(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

    const selectedModel = models.find((m) => m.name === selected);
    const canLoad = !!selectedModel && props.status !== "loading";

    // Compact loading badge — "% done · ETA Xm Ys". The full
    // byte-counter + rate string lives in the progress bar that sits
    // under the page header, no need to repeat it here.
    const loadingCompact = (() => {
        if (props.status !== "loading") return null;
        const done = Math.max(0, Math.min(100, props.loadingPercent));
        const m    = /ETA\s+(.+)$/.exec(props.loadingLabel);
        const eta  = m ? m[1] : null;
        return eta
            ? `${done.toFixed(0)}% · ETA ${eta}`
            : done > 0
                ? `${done.toFixed(0)}%`
                : "loading…";
    })();

    return (
        <div className="flex flex-col gap-1">
            {/* Row 1: controls — model picker + Load + delete. min-w-0 on
                the select lets it shrink so the action buttons keep their
                fixed widths on narrow sidebars instead of wrapping. */}
            <div className="flex flex-nowrap items-center gap-1">
                <select
                    value={selected}
                    onChange={(e) => setSelected(e.target.value)}
                    disabled={refreshing || models.length === 0 || props.status === "loading"}
                    className="h-7 min-w-0 flex-1 rounded border border-input bg-background px-2 text-xs shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:opacity-60"
                    title={props.status === "loading" ? "Locked while a model is downloading" : "Choose a model to load"}
                >
                    {models.length === 0 && <option value="">— scanning —</option>}
                    {models.map((m) => (
                        <option key={m.name} value={m.name}>
                            {isSupported(m) ? "✓ " : "✗ "} {m.name} — {fmtBytes(m.size)}{m.heavy ? " ⚠" : ""}
                        </option>
                    ))}
                </select>
                {props.status === "loading" && props.onCancel ? (
                    <AlertDialog>
                        <AlertDialogTrigger asChild>
                            <Button
                                variant="destructive"
                                size="sm"
                                className="h-7 shrink-0 px-2 text-xs"
                                title="Stop the download"
                            >
                                <Square />
                                Stop
                            </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                            <AlertDialogHeader>
                                <AlertDialogTitle>Stop download?</AlertDialogTitle>
                                <AlertDialogDescription>
                                    {loadingCompact ? `${loadingCompact} downloaded. ` : ""}
                                    Keep the partial in this browser's storage to resume
                                    later, or delete it to free the space now.
                                </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                                <AlertDialogCancel>Continue downloading</AlertDialogCancel>
                                {props.onCancelDelete && (
                                    <AlertDialogAction
                                        onClick={() => props.onCancelDelete?.()}
                                        className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                    >
                                        Delete partial
                                    </AlertDialogAction>
                                )}
                                <AlertDialogAction onClick={() => props.onCancel?.()}>
                                    Keep partial
                                </AlertDialogAction>
                            </AlertDialogFooter>
                        </AlertDialogContent>
                    </AlertDialog>
                ) : (
                    <Button
                        size="sm"
                        className="h-7 shrink-0 px-2 text-xs"
                        onClick={() => selectedModel && props.onLoad(selectedModel)}
                        disabled={!canLoad}
                    >
                        <Download />
                        Load
                    </Button>
                )}
                {props.onDelete && (
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 shrink-0 text-muted-foreground hover:text-destructive"
                        onClick={() => selectedModel && props.onDelete?.(selectedModel)}
                        disabled={!selectedModel || props.status === "loading"}
                        title="Delete cached model from OPFS"
                    >
                        <Trash2 />
                    </Button>
                )}
            </div>

            {/* Advisory caution for heavy models (gemma4:12b dense, gemma4:26b
                MoE). Load stays enabled on every tier — this only warns. */}
            {selectedModel?.heavy && (
                <p className="flex items-start gap-1 text-[11px] leading-tight text-amber-600 dark:text-amber-500">
                    <AlertTriangle className="mt-px size-3 shrink-0" />
                    <span>
                        Very heavy model — a large download, slow on modest hardware. The 26B
                        sparse-MoE streams its experts to fit low-VRAM GPUs (a few seconds per
                        token); dense models like 12B want more VRAM.
                    </span>
                </p>
            )}

            {/* Row 2: status — only renders when there's something to say. */}
            {(props.status === "ready" || props.status === "loading" || props.status === "error" || error) && (
                <div className="flex flex-wrap items-center gap-1">
                    {props.status === "ready" && (
                        <>
                            <Badge tone="ok" className="truncate max-w-[14rem]" title={props.statusText}>
                                {props.statusText}
                            </Badge>
                            {props.onEject && (
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-7 w-7 text-muted-foreground hover:text-foreground"
                                    onClick={() => props.onEject?.()}
                                    title="Eject model (clears auto-load on reload)"
                                >
                                    <Unplug />
                                </Button>
                            )}
                        </>
                    )}
                    {props.status === "loading" && loadingCompact && (
                        <Badge tone="warn" className="font-mono tabular-nums" title={props.loadingLabel}>
                            {loadingCompact}
                        </Badge>
                    )}
                    {props.status === "error" && <Badge tone="err">error</Badge>}
                    {error && <span className="text-xs text-destructive">{error}</span>}
                </div>
            )}
        </div>
    );
}

/** Thin progress bar shown under the toolbar while a model is loading. */
export function ModelLoadProgress(props: { percent: number; label: string }) {
    return (
        <div className="space-y-0.5 border-b border-border bg-background/50 px-3 py-1">
            <Progress value={props.percent} />
            <p className="font-mono tabular-nums text-[0.65rem] text-muted-foreground">
                {props.label}
            </p>
        </div>
    );
}
