import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { fmtBytes } from "@/lib/utils";
import { type ModelEntry, isSupported, listModels } from "@/lib/api";
import { Download, Trash2, Unplug } from "lucide-react";

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
    onCancel?: () => void;
}

/** Compact single-row model picker. Sits in the top toolbar. */
export function ModelLoader(props: Props) {
    const [models, setModels] = useState<ModelEntry[]>([]);
    const [selected, setSelected] = useState<string>("");
    const [refreshing, setRefreshing] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const refresh = async () => {
        setRefreshing(true);
        setError(null);
        try {
            const ms = await listModels();
            setModels(ms);
            if (ms.length > 0 && !selected) {
                const firstSupported = ms.find(isSupported);
                setSelected((firstSupported ?? ms[0]).name);
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
                    disabled={refreshing || models.length === 0}
                    className="h-7 min-w-0 flex-1 rounded border border-input bg-background px-2 text-xs shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    title="Choose a model to load"
                >
                    {models.length === 0 && <option value="">— scanning —</option>}
                    {models.map((m) => (
                        <option key={m.name} value={m.name}>
                            {isSupported(m) ? "✓ " : "✗ "} {m.name} — {fmtBytes(m.size)}
                        </option>
                    ))}
                </select>
                <Button
                    size="sm"
                    className="h-7 shrink-0 px-2 text-xs"
                    onClick={() => selectedModel && props.onLoad(selectedModel)}
                    disabled={!canLoad}
                >
                    <Download />
                    Load
                </Button>
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
