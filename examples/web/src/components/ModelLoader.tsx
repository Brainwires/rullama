import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
        <Card>
            <CardHeader>
                <CardTitle>Model</CardTitle>
                <CardDescription>
                    Pick a model from your local <code>~/.ollama/models</code>. Only{" "}
                    <code>gemma4:*</code> variants will run; others fail at parse time.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                    <select
                        value={selected}
                        onChange={(e) => setSelected(e.target.value)}
                        disabled={refreshing || models.length === 0}
                        className="h-9 min-w-[12rem] flex-1 rounded-md border border-input bg-background px-3 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring sm:flex-initial"
                    >
                        {models.length === 0 && <option value="">— scanning —</option>}
                        {models.map((m) => (
                            <option key={m.name} value={m.name}>
                                {isSupported(m) ? "✓ " : "✗ "} {m.name} — {fmtBytes(m.size)}
                            </option>
                        ))}
                    </select>
                    <Button variant="outline" size="icon" onClick={refresh} disabled={refreshing} title="Refresh model list">
                        <RefreshCw className={refreshing ? "animate-spin" : ""} />
                    </Button>
                    <Button onClick={() => selectedModel && props.onLoad(selectedModel)} disabled={!canLoad}>
                        <Download />
                        Load
                    </Button>
                    {props.status === "ready" && <Badge tone="ok">{props.statusText}</Badge>}
                    {props.status === "error" && <Badge tone="err">{props.statusText}</Badge>}
                    {props.status === "loading" && <Badge tone="warn">{props.statusText || "loading…"}</Badge>}
                </div>

                {props.status === "loading" && (
                    <div className="space-y-1">
                        <Progress value={props.loadingPercent} />
                        <p className="text-xs text-muted-foreground">{props.loadingLabel}</p>
                    </div>
                )}

                {error && <p className="text-xs text-destructive">{error}</p>}
            </CardContent>
        </Card>
    );
}
