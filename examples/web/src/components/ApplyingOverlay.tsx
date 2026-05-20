import { Loader2 } from "lucide-react";
import { shortVersionLabel } from "@/lib/version";

interface Props {
    /** Server-side version being applied. */
    version: string;
}

/** Full-screen overlay shown on every tab during the coordinated update
 *  reload. Mirrors the visual language of the static-HTML splash in
 *  index.html so the transition from "in-app" → "boot" feels seamless
 *  to the user.
 *
 *  Rendered for ~600 ms before App.tsx fires `window.location.reload()`
 *  — enough time for the dedicated core worker to process the shutdown
 *  message and release its OPFS sync handle + GPU towers before the
 *  new bundle's worker tries to acquire them. */
export function ApplyingOverlay({ version }: Props) {
    return (
        <div
            className="fixed inset-0 z-[99997] flex flex-col items-center justify-center gap-3 bg-background/95 backdrop-blur-sm px-6 text-center"
            role="alert"
            aria-live="assertive"
        >
            <img
                src="/icons/icon-512.png"
                alt=""
                aria-hidden="true"
                className="h-24 w-24 rounded-2xl shadow-lg"
            />
            <div className="flex items-center gap-2 text-sm font-medium">
                <Loader2 className="h-4 w-4 animate-spin text-primary" aria-hidden="true" />
                Updating rullama…
            </div>
            <p className="max-w-xs text-xs text-muted-foreground">
                Installing <span className="font-mono">{shortVersionLabel(version)}</span> and
                reloading. The cached model stays in place — no redownload.
            </p>
        </div>
    );
}
