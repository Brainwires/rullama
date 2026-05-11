import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface EnvCheck {
    webgpu: boolean;
    secureContext: boolean;
    crossOriginIsolated: boolean;
    opfs: boolean;
}

function probeEnv(): EnvCheck {
    return {
        webgpu: typeof navigator !== "undefined" && "gpu" in navigator,
        secureContext: typeof window !== "undefined" && window.isSecureContext,
        crossOriginIsolated: typeof window !== "undefined" && window.crossOriginIsolated,
        opfs:
            typeof navigator !== "undefined" &&
            !!navigator.storage &&
            typeof navigator.storage.getDirectory === "function",
    };
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
    return (
        <span
            className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium",
                ok
                    ? "bg-green-500/15 text-green-400 ring-1 ring-green-500/30"
                    : "bg-destructive/15 text-destructive ring-1 ring-destructive/30",
            )}
        >
            <span className={cn("h-1.5 w-1.5 rounded-full", ok ? "bg-green-400" : "bg-destructive")} />
            {label}: {ok ? "ok" : "no"}
        </span>
    );
}

export function App() {
    const [env, setEnv] = useState<EnvCheck>({
        webgpu: false,
        secureContext: false,
        crossOriginIsolated: false,
        opfs: false,
    });

    useEffect(() => {
        setEnv(probeEnv());
    }, []);

    return (
        <div className="min-h-screen bg-background safe-top">
            <div className="mx-auto max-w-3xl px-4 py-6 sm:py-10">
                <header className="mb-6">
                    <h1 className="text-2xl font-semibold tracking-tight">rullama</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        Gemma 4 in your browser. Pure Rust → WebAssembly + WebGPU. No server.
                    </p>
                </header>

                <section className="rounded-lg border border-border bg-card p-4">
                    <h2 className="mb-3 text-sm font-medium">Environment</h2>
                    <div className="flex flex-wrap gap-2">
                        <StatusPill ok={env.webgpu} label="WebGPU" />
                        <StatusPill ok={env.secureContext} label="Secure" />
                        <StatusPill ok={env.crossOriginIsolated} label="COOP/COEP" />
                        <StatusPill ok={env.opfs} label="OPFS" />
                    </div>
                    <p className="mt-3 text-xs text-muted-foreground">
                        WebGPU is required to run inference. OPFS is required to load models &gt; 4 GB.
                    </p>
                </section>

                <section className="mt-6 rounded-lg border border-dashed border-border bg-card/30 p-6 text-center">
                    <p className="text-sm text-muted-foreground">
                        UI scaffolding in progress — model loader + chat panel ship next.
                    </p>
                </section>
            </div>
        </div>
    );
}
