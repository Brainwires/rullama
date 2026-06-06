import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";

interface EnvCheck {
    webgpu: boolean;
    secureContext: boolean;
    crossOriginIsolated: boolean;
    opfs: boolean;
}

function probeEnv(): EnvCheck {
    return {
        webgpu:              typeof navigator !== "undefined" && "gpu" in navigator,
        secureContext:       typeof window    !== "undefined" && window.isSecureContext,
        crossOriginIsolated: typeof window    !== "undefined" && window.crossOriginIsolated,
        opfs:                typeof navigator !== "undefined"
                                && !!navigator.storage
                                && typeof navigator.storage.getDirectory === "function",
    };
}

/** Inline strip of environment-support pills. No card, no header. */
export function EnvironmentStatus() {
    const [env, setEnv] = useState<EnvCheck | null>(null);

    useEffect(() => { setEnv(probeEnv()); }, []);

    return (
        <div
            className="flex flex-wrap items-center gap-1"
            title={
                env
                    ? `WebGPU=${env.webgpu} · Secure=${env.secureContext} · COI=${env.crossOriginIsolated} · OPFS=${env.opfs}`
                    : "probing environment…"
            }
        >
            <Badge tone={env?.webgpu              ? "ok" : "err"}>GPU</Badge>
            <Badge tone={env?.opfs                ? "ok" : "err"}>OPFS</Badge>
            <Badge tone={env?.crossOriginIsolated ? "ok" : "warn"}>COI</Badge>
        </div>
    );
}
