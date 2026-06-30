// Full-page overlay that asks the user to restart after a deploy made the
// running tab's JS / wasm references go stale. See `lib/restart.ts` for the
// signal mechanics — this is purely the UI side.
//
// Z-index sits above every other layer (modals, drawers, toasts). The
// overlay is dismiss-proof on purpose: once we know the wasm/SW is in a
// broken state, every other interaction will fail too, so giving the user
// "click the Restart button" is the only useful action.

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useNeedsRestart, restartNow } from "@/lib/restart";
import { RefreshCw } from "lucide-react";

export function RestartOverlay() {
    const reason = useNeedsRestart();
    if (!reason) return null;

    return (
        <div
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="restart-overlay-title"
            className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4 safe-top safe-bottom"
        >
            <Card className="w-full max-w-sm shadow-2xl animate-fade-in">
                <CardHeader>
                    <CardTitle id="restart-overlay-title" className="flex items-center gap-2">
                        <RefreshCw className="h-5 w-5" />
                        Restart required
                    </CardTitle>
                    <CardDescription>
                        rullama was updated — {reason}. A restart is required to
                        load the new WebAssembly bundle.
                    </CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-2">
                    <Button
                        onClick={() => restartNow()}
                        autoFocus
                        className="w-full"
                    >
                        Restart now
                    </Button>
                    <p className="text-[0.65rem] text-muted-foreground">
                        In-progress generation will be lost. Past
                        conversations and downloaded models stay cached.
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}
