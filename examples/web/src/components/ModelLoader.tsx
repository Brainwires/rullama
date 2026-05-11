import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { fmtBytes } from "@/lib/utils";
import { type ModelEntry, isSupported, listModels } from "@/lib/api";
import { RefreshCw, Download } from "lucide-react";

export type ModelStatus = "idle" | "loading" | "ready" | "error";

interface Props {
    status: ModelStatus;
    loadingPercent: number;     // 0..100 during download/load
    loadingLabel: string;       // e.g. "5.4 GB / 7.16 GB — 81.2 MB/s"
    statusText: string;         // e.g. "ready: gemma4:e2b"
    onLoad: (model: ModelEntry) => void;
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

    return (
        <div className="flex flex-wrap items-center gap-1">
            <select
                value={selected}
                onChange={(e) => setSelected(e.target.value)}
                disabled={refreshing || models.length === 0}
                className="h-7 max-w-[12rem] rounded border border-input bg-background px-2 text-xs shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                title="Choose a model from ~/.ollama/models"
            >
                {models.length === 0 && <option value="">— scanning —</option>}
                {models.map((m) => (
                    <option key={m.name} value={m.name}>
                        {isSupported(m) ? "✓ " : "✗ "} {m.name} — {fmtBytes(m.size)}
                    </option>
                ))}
            </select>
            <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={refresh}
                disabled={refreshing}
                title="Refresh model list"
            >
                <RefreshCw className={refreshing ? "animate-spin" : ""} />
            </Button>
            <Button
                size="sm"
                className="h-7 px-2 text-xs"
                onClick={() => selectedModel && props.onLoad(selectedModel)}
                disabled={!canLoad}
            >
                <Download />
                Load
            </Button>
            {props.status === "ready"   && (
                <Badge tone="ok"   className="truncate max-w-[14rem]" title={props.statusText}>
                    {props.statusText}
                </Badge>
            )}
            {props.status === "error"   && <Badge tone="err">error</Badge>}
            {props.status === "loading" && (
                <Badge tone="warn" className="truncate max-w-[14rem]" title={props.loadingLabel}>
                    {props.loadingLabel || "loading…"}
                </Badge>
            )}
            {error && <span className="text-xs text-destructive">{error}</span>}
        </div>
    );
}

/** Thin progress bar shown under the toolbar while a model is loading. */
export function ModelLoadProgress(props: { percent: number; label: string }) {
    return (
        <div className="space-y-0.5 border-b border-border bg-background/50 px-3 py-1">
            <Progress value={props.percent} />
            <p className="text-[0.65rem] text-muted-foreground">{props.label}</p>
        </div>
    );
}
