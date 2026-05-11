import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface EnvCheck {
    webgpu: boolean;
    secureContext: boolean;
    crossOriginIsolated: boolean;
    opfs: boolean;
    workerSyncAccess: boolean;
}

function probeEnv(): EnvCheck {
    return {
        webgpu:              typeof navigator !== "undefined" && "gpu" in navigator,
        secureContext:       typeof window    !== "undefined" && window.isSecureContext,
        crossOriginIsolated: typeof window    !== "undefined" && window.crossOriginIsolated,
        opfs:                typeof navigator !== "undefined"
                                && !!navigator.storage
                                && typeof navigator.storage.getDirectory === "function",
        // SyncAccessHandle is only callable inside a Worker but its
        // existence on FileSystemFileHandle is a soft hint that OPFS+sync
        // is supported. We don't strictly check it here.
        workerSyncAccess:    typeof FileSystemFileHandle !== "undefined",
    };
}

export function EnvironmentStatus() {
    const [env, setEnv] = useState<EnvCheck | null>(null);

    useEffect(() => { setEnv(probeEnv()); }, []);

    return (
        <Card>
            <CardHeader>
                <CardTitle>Environment</CardTitle>
                <CardDescription>
                    WebGPU is required for inference. OPFS is required for models &gt; 4 GB.
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="flex flex-wrap gap-2">
                    <Badge tone={env?.webgpu               ? "ok" : "err"}>WebGPU</Badge>
                    <Badge tone={env?.secureContext        ? "ok" : "err"}>Secure</Badge>
                    <Badge tone={env?.crossOriginIsolated  ? "ok" : "warn"}>COOP/COEP</Badge>
                    <Badge tone={env?.opfs                 ? "ok" : "err"}>OPFS</Badge>
                </div>
            </CardContent>
        </Card>
    );
}
