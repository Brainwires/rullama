// Image tab — text-to-image generation with Z-Image-Turbo (the 4th engine,
// the `ImageModel` wasm class). Unlike the chat models this loads from a CDN
// base URL via HTTP Range (nothing touches OPFS) and runs the whole pipeline
// in one async call. The engine reports fine-grained progress (per encoder/DiT
// layer + per VAE stage) via an `imageStep` notify, so the UI shows a live
// phase label + progress bar rather than an opaque busy state.
//
// Lifecycle is tab-owned: the component loads the engine via
// `client.image.load` (probing existing status on mount) and unloads it on
// unmount. App.tsx also tears the inference core down when leaving the tab on
// non-premium devices, so a stale GPU resident is reclaimed either way.

import { useCallback, useEffect, useRef, useState } from "react";
import { ImageIcon, Loader2, Download, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { getClient } from "@/lib/inference";
import { IMAGE_MODEL } from "@/lib/api";

type EngineStatus = "idle" | "loading" | "ready" | "error";

// Offered output sizes. `lh`/`lw` are LATENT dims (image px ÷ 8).
const SIZES = [
    { label: "256 × 256", px: 256 },
    { label: "512 × 512", px: 512 },
] as const;

export function ImageTab() {
    const client = getClient();
    const [status, setStatus] = useState<EngineStatus>("idle");
    const [err, setErr] = useState<string | null>(null);

    const [prompt, setPrompt] = useState("");
    const [negPrompt, setNegPrompt] = useState("");
    const [sizePx, setSizePx] = useState<number>(512);
    const [steps, setSteps] = useState<number>(9);
    const [cfg, setCfg] = useState<number>(4.0);
    const [seed, setSeed] = useState<number>(42);

    const [busy, setBusy] = useState(false);
    const [haveImage, setHaveImage] = useState(false);
    // Live progress relayed from the engine (per encoder/DiT layer, per VAE
    // stage) so the user sees constant movement, not a silent multi-minute hang.
    const [progress, setProgress] = useState<{ label: string; done: number; total: number } | null>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Subscribe to fine-grained generation progress while mounted.
    useEffect(() => {
        const unsub = client.image.onStep((p) => setProgress(p));
        return () => { unsub(); };
    }, [client]);

    // Reflect existing engine status on mount; the worker is the source of
    // truth (another tab may have loaded it already).
    useEffect(() => {
        let alive = true;
        (async () => {
            const s = await client.image.status().catch(() => null);
            if (alive && s) setStatus("ready");
        })();
        return () => { alive = false; };
    }, [client]);

    const onLoad = useCallback(async () => {
        setErr(null);
        setStatus("loading");
        try {
            await client.image.load(IMAGE_MODEL.baseUrl, IMAGE_MODEL.name);
            setStatus("ready");
        } catch (e) {
            setStatus("error");
            setErr((e as Error).message ?? String(e));
        }
    }, [client]);

    const onUnload = useCallback(async () => {
        try { await client.image.unload(); } catch { /* */ }
        setStatus("idle");
        setHaveImage(false);
    }, [client]);

    // Tear the engine down when the tab unmounts so its GPU buffers are freed.
    useEffect(() => {
        return () => { void client.image.unload().catch(() => {}); };
    }, [client]);

    const onGenerate = useCallback(async () => {
        if (status !== "ready" || busy || !prompt.trim()) return;
        setBusy(true);
        setErr(null);
        setProgress({ label: "Starting…", done: 0, total: 1 });
        try {
            const latent = Math.round(sizePx / 8);
            const { rgba, width, height } = await client.image.generate({
                prompt: prompt.trim(),
                negPrompt: negPrompt.trim(),
                lh: latent,
                lw: latent,
                steps,
                cfgScale: cfg,
                seed,
            });
            const canvas = canvasRef.current;
            if (canvas) {
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext("2d");
                if (ctx) {
                    const bytes = rgba instanceof Uint8Array ? rgba : new Uint8Array(rgba);
                    ctx.putImageData(
                        new ImageData(new Uint8ClampedArray(bytes), width, height),
                        0, 0,
                    );
                    setHaveImage(true);
                }
            }
        } catch (e) {
            setErr((e as Error).message ?? String(e));
        } finally {
            setBusy(false);
            setProgress(null);
        }
    }, [client, status, busy, prompt, negPrompt, sizePx, steps, cfg, seed]);

    const onDownload = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas || !haveImage) return;
        canvas.toBlob((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `z-image-${Date.now()}.png`;
            a.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        }, "image/png");
    }, [haveImage]);

    return (
        <div className="mx-auto flex h-full w-full max-w-5xl min-h-0 flex-col gap-4 overflow-y-auto p-4">
            <div className="flex items-center gap-2 text-lg font-semibold">
                <ImageIcon className="size-5" />
                <span>Image generation</span>
                <span className="text-xs font-normal text-muted-foreground">
                    Z-Image-Turbo · runs entirely on your GPU
                </span>
            </div>

            {/* Engine load / status. ~31 GB; streamed per-tensor from the CDN. */}
            {status !== "ready" && (
                <Card>
                    <CardContent className="flex flex-col gap-3 p-4">
                        <p className="text-sm text-muted-foreground">
                            The image engine streams ~31 GB of weights from the CDN per-tensor
                            (nothing is cached to disk). Desktop GPU only — generation is slow
                            on integrated graphics.
                        </p>
                        <div className="flex items-center gap-2">
                            <Button onClick={() => void onLoad()} disabled={status === "loading"}>
                                {status === "loading" ? (
                                    <><Loader2 className="mr-1 size-4 animate-spin" /> Loading engine…</>
                                ) : (
                                    <>Load image engine</>
                                )}
                            </Button>
                        </div>
                        {err && <p className="text-sm text-destructive">{err}</p>}
                    </CardContent>
                </Card>
            )}

            {status === "ready" && (
                <>
                    <Card>
                        <CardContent className="flex flex-col gap-3 p-4">
                            <label className="text-sm font-medium">Prompt</label>
                            <Textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder="A photorealistic red fox sitting in autumn leaves, golden hour"
                                rows={3}
                                disabled={busy}
                            />
                            <label className="text-sm font-medium">Negative prompt (optional)</label>
                            <Input
                                value={negPrompt}
                                onChange={(e) => setNegPrompt(e.target.value)}
                                placeholder="blurry, low quality, watermark"
                                disabled={busy}
                            />

                            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                                <div className="flex flex-col gap-1">
                                    <label className="text-xs text-muted-foreground">Size</label>
                                    <div className="flex gap-1">
                                        {SIZES.map((s) => (
                                            <button
                                                key={s.px}
                                                type="button"
                                                disabled={busy}
                                                onClick={() => setSizePx(s.px)}
                                                className={cn(
                                                    "flex-1 rounded border px-2 py-1 text-xs transition-colors",
                                                    sizePx === s.px
                                                        ? "border-primary bg-primary/10 text-foreground"
                                                        : "border-border text-muted-foreground hover:bg-muted/50",
                                                )}
                                            >
                                                {s.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div className="flex flex-col gap-1">
                                    <label className="text-xs text-muted-foreground">Steps</label>
                                    <Input
                                        type="number"
                                        min={1}
                                        max={50}
                                        value={steps}
                                        onChange={(e) => setSteps(Math.max(1, Number(e.target.value) || 1))}
                                        disabled={busy}
                                    />
                                </div>
                                <div className="flex flex-col gap-1">
                                    <label className="text-xs text-muted-foreground">CFG scale</label>
                                    <Input
                                        type="number"
                                        min={0}
                                        max={20}
                                        step={0.5}
                                        value={cfg}
                                        onChange={(e) => setCfg(Math.max(0, Number(e.target.value) || 0))}
                                        disabled={busy}
                                    />
                                </div>
                                <div className="flex flex-col gap-1">
                                    <label className="text-xs text-muted-foreground">Seed</label>
                                    <Input
                                        type="number"
                                        value={seed}
                                        onChange={(e) => setSeed(Number(e.target.value) || 0)}
                                        disabled={busy}
                                    />
                                </div>
                            </div>

                            <div className="flex items-center gap-2">
                                <Button onClick={() => void onGenerate()} disabled={busy || !prompt.trim()}>
                                    {busy ? (
                                        <><Loader2 className="mr-1 size-4 animate-spin" /> Generating…</>
                                    ) : (
                                        <><Sparkles className="mr-1 size-4" /> Generate</>
                                    )}
                                </Button>
                                <Button variant="ghost" onClick={() => void onUnload()} disabled={busy}>
                                    Eject engine
                                </Button>
                            </div>
                            {err && <p className="text-sm text-destructive">{err}</p>}
                            {/* Live progress: the engine reports per encoder/DiT
                                layer + per VAE stage. Generation is network-bound
                                (each DiT step re-streams ~10 GB of weights), so
                                this moves steadily but the whole run is slow. */}
                            {busy && (
                                <div className="flex flex-col gap-1.5">
                                    <p className="flex items-center gap-2 text-xs text-muted-foreground">
                                        <Loader2 className="size-3 animate-spin" />
                                        {progress?.label ?? "Running pipeline…"}
                                        {progress && progress.total > 1 && (
                                            <span className="tabular-nums">
                                                {progress.done + 1}/{progress.total}
                                            </span>
                                        )}
                                    </p>
                                    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                                        <div
                                            className="h-full bg-primary transition-[width] duration-200"
                                            style={{
                                                width: progress && progress.total > 0
                                                    ? `${Math.min(100, ((progress.done + 1) / progress.total) * 100)}%`
                                                    : "10%",
                                            }}
                                        />
                                    </div>
                                    <p className="text-[10px] text-muted-foreground">
                                        Weights stream from the CDN per step — this is network-bound, so
                                        expect minutes per step on a slow connection.
                                    </p>
                                </div>
                            )}
                        </CardContent>
                    </Card>

                    <Card>
                        <CardContent className="flex flex-col items-center gap-3 p-4">
                            <div className="relative flex w-full items-center justify-center rounded bg-muted/30 p-2">
                                <canvas
                                    ref={canvasRef}
                                    className={cn(
                                        "max-h-[60vh] max-w-full rounded",
                                        haveImage ? "block" : "hidden",
                                    )}
                                />
                                {!haveImage && (
                                    <div className="flex flex-col items-center gap-2 py-16 text-muted-foreground">
                                        <ImageIcon className="size-8" />
                                        <span className="text-sm">Your generated image will appear here</span>
                                    </div>
                                )}
                            </div>
                            {haveImage && (
                                <Button variant="outline" onClick={onDownload}>
                                    <Download className="mr-1 size-4" /> Download PNG
                                </Button>
                            )}
                        </CardContent>
                    </Card>
                </>
            )}
        </div>
    );
}
