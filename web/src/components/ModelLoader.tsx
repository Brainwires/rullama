import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
    Select, SelectContent, SelectItem, SelectTrigger,
} from "@/components/ui/select";
import { fmtBytes } from "@/lib/utils";
import { type ModelEntry, isCloud, isSupported, listModels } from "@/lib/api";
import { hasCloudKey } from "@/lib/cloud/keyvault";
import { providerLabel } from "@/lib/cloud/types";
import { existingSize } from "@/lib/opfs";
import {
    AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
    AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
    AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { AlertTriangle, Clock, Cloud, Download, HardDrive, Loader2, Square, Trash2, Unplug } from "lucide-react";

/** Download state of a catalog model, shown as an icon in the picker. */
type CacheState = "cached" | "downloading" | "available" | "cloud";

function StatusIcon({ state }: { state: CacheState }) {
    const base = "size-3.5 shrink-0";
    if (state === "cached")
        return <HardDrive className={`${base} text-emerald-500`} aria-label="Downloaded (in this browser)" />;
    if (state === "downloading")
        return <Clock className={`${base} text-amber-500`} aria-label="Downloading" />;
    if (state === "cloud")
        return <Cloud className={`${base} text-sky-500`} aria-label="Cloud model (runs on the provider's servers)" />;
    return <Cloud className={`${base} text-muted-foreground`} aria-label="Available to download" />;
}

export type ModelStatus = "idle" | "loading" | "preparing" | "ready" | "error";

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
    // True when the selected model is a cloud model whose provider API key
    // isn't set yet — Load stays blocked with an inline "set key" hint.
    const [cloudKeyMissing, setCloudKeyMissing] = useState(false);

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

    // Per-model "fully downloaded into OPFS?" map, for the picker's status icon.
    // Recomputed whenever the catalog changes or a load finishes (so a model
    // flips to the local-storage icon the moment its download completes).
    const [cached, setCached] = useState<Record<string, boolean>>({});
    useEffect(() => {
        let stop = false;
        void (async () => {
            const entries = await Promise.all(models.map(async (m) => {
                if (isCloud(m)) return [m.name, false] as const; // never OPFS-cached
                const modelKey = m.digest.replace(/[^A-Za-z0-9_.-]/g, "_");
                const filename = m.name.replace(/[^A-Za-z0-9_.-]/g, "_") + ".gguf";
                try {
                    const sz = await existingSize(modelKey, filename);
                    return [m.name, m.size > 0 && sz >= m.size] as const;
                } catch { return [m.name, false] as const; }
            }));
            if (!stop) setCached(Object.fromEntries(entries));
        })();
        return () => { stop = true; };
    }, [models, props.status]);

    const stateFor = (m: ModelEntry): CacheState => {
        if (isCloud(m)) return "cloud";
        if (props.status === "loading" && m.name === selected) return "downloading";
        return cached[m.name] ? "cached" : "available";
    };

    const selectedModel = models.find((m) => m.name === selected);

    // When a cloud model is selected, check whether its provider key is set so
    // we can block Load + show a hint. Local models clear the flag.
    useEffect(() => {
        let stop = false;
        const cloud = selectedModel?.cloud;
        if (!cloud) { setCloudKeyMissing(false); return; }
        void hasCloudKey(cloud.provider).then((present) => {
            if (!stop) setCloudKeyMissing(!present);
        });
        return () => { stop = true; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selected, models, props.status]);

    const cloudBlocked = !!selectedModel?.cloud && cloudKeyMissing;
    const canLoad = !!selectedModel && props.status !== "loading" && !cloudBlocked;

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
                <Select
                    value={selected}
                    onValueChange={setSelected}
                    disabled={refreshing || models.length === 0 || props.status === "loading"}
                >
                    <SelectTrigger
                        className="h-7 min-w-0 flex-1"
                        title={props.status === "loading" ? "Locked while a model is downloading" : "Choose a model to load"}
                    >
                        {selectedModel ? (
                            <span className="flex min-w-0 items-center gap-1.5">
                                <StatusIcon state={stateFor(selectedModel)} />
                                <span className="truncate">{selectedModel.name}</span>
                            </span>
                        ) : (
                            <span className="text-muted-foreground">{models.length === 0 ? "— scanning —" : "Select a model"}</span>
                        )}
                    </SelectTrigger>
                    <SelectContent>
                        {models.map((m) => (
                            <SelectItem key={m.name} value={m.name} disabled={!isSupported(m)}>
                                <span className="flex items-center gap-1.5">
                                    <StatusIcon state={stateFor(m)} />
                                    <span>{m.name} — {m.cloud ? providerLabel(m.cloud.provider) : fmtBytes(m.size)}</span>
                                    {m.heavy && <AlertTriangle className="size-3 shrink-0 text-amber-500" />}
                                </span>
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
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

            {/* Cloud privacy note + key-missing hint. The model runs on the
                provider's servers, so flag that data leaves the device; if the
                BYOK key isn't set, point the user at Settings and block Load. */}
            {selectedModel?.cloud && (
                <p className={`flex items-start gap-1 text-[11px] leading-tight ${
                    cloudKeyMissing ? "text-amber-600 dark:text-amber-500" : "text-sky-600 dark:text-sky-400"
                }`}>
                    <Cloud className="mt-px size-3 shrink-0" />
                    <span>
                        Runs on {providerLabel(selectedModel.cloud.provider)}'s servers — your
                        messages leave this device.
                        {cloudKeyMissing && <> Add your API key in <strong>Settings → Cloud</strong> to use it.</>}
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

/** Active-load view: replaces the model picker while a model is
 *  downloading/loading ("loading") or pre-warming the system prompt
 *  ("preparing"). Shows a spinner + progress bar + label, and a Stop
 *  control during the download phase. */
export function LoadingModel(props: {
    status: ModelStatus;            // "loading" | "preparing"
    percent: number;
    label: string;
    modelName?: string;
    /** Stop the download, keep the partial in OPFS (download phase only). */
    onCancel?: () => void;
    /** Stop the download AND delete the partial (download phase only). */
    onCancelDelete?: () => void;
}) {
    const isPreparing = props.status === "preparing";
    const pct = Math.max(0, Math.min(100, props.percent));
    const name = props.modelName?.trim();
    const heading = isPreparing
        ? `Preparing ${name ?? "model"}…`
        : `Loading ${name ?? "model"}…`;
    return (
        <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
                <Loader2 className="size-4 shrink-0 animate-spin text-primary" />
                <span className="truncate text-sm font-medium" title={heading}>{heading}</span>
            </div>
            <Progress value={pct} />
            <p className="font-mono tabular-nums text-[0.7rem] text-muted-foreground">
                {props.label || (isPreparing ? "preparing…" : "starting…")}
            </p>
            {/* Stop is meaningful only during the download/load phase; the
                short prepare prefill has nothing to cancel. */}
            {!isPreparing && props.onCancel && (
                <AlertDialog>
                    <AlertDialogTrigger asChild>
                        <Button
                            variant="destructive"
                            size="sm"
                            className="h-7 w-fit px-2 text-xs"
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
