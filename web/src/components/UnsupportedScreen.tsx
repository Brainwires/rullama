import { AlertTriangle } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { GpuProbe } from "@/lib/capability";

/**
 * Full-page block shown when the device is below the minimum spec
 * (no WebGPU, or a GPU too small to hold a 512 MB weight tile). Rendered at
 * the top of App BEFORE the engine/worker boots, so incapable devices
 * (e.g. iPhone 7 — no WebGPU) can't boot-loop the heavy path.
 */
export function UnsupportedScreen({ probe }: { probe: GpuProbe | null }) {
    const reason = !probe?.hasGpu
        ? "This browser/device doesn't expose WebGPU (`navigator.gpu`), which rullama needs to run the model on the GPU. Older iPhones (pre-A-series WebGPU) and unsupported browsers land here."
        : !probe.adapterOk
            ? `WebGPU is present but no GPU adapter is available${probe.error ? ` (${probe.error})` : ""}. The GPU driver may be denying access, or there's no compatible GPU.`
            : `Your GPU advertises a max single-buffer of ${Math.round((probe.maxBufferSize ?? 0) / 1024 / 1024)} MB. rullama needs at least 512 MB to hold a weight tile.`;

    return (
        <div className="flex min-h-screen items-center justify-center bg-background p-6">
            <Card className="max-w-lg">
                <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                        <AlertTriangle className="size-4 text-amber-500" />
                        This device doesn't meet the minimum requirements
                    </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                    <p className="text-sm text-muted-foreground">{reason}</p>
                    <p className="text-xs text-muted-foreground">
                        Minimum: an iPhone&nbsp;16e-class GPU or newer / a desktop browser with WebGPU
                        (Chrome&nbsp;113+, Edge&nbsp;113+, or recent Safari). Recommended for desktop: a
                        12&nbsp;GB GPU; 24&nbsp;GB+ for the largest models.
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}
