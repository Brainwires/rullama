import { useToast, type ToastLevel } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { X, AlertTriangle, CircleAlert, Info, CheckCircle2 } from "lucide-react";

const STYLE: Record<ToastLevel, { border: string; bg: string; icon: typeof Info }> = {
    error:   { border: "border-l-destructive",   bg: "bg-destructive/10",      icon: CircleAlert },
    warn:    { border: "border-l-yellow-500",    bg: "bg-yellow-500/10",       icon: AlertTriangle },
    success: { border: "border-l-emerald-500",   bg: "bg-emerald-500/10",      icon: CheckCircle2 },
    info:    { border: "border-l-muted-foreground", bg: "bg-muted/40",         icon: Info },
};

/** Bottom-right toast stack. Persistent (error/warn) toasts stay until
 *  dismissed; info/success auto-dismiss in the provider. */
export function Toaster() {
    const { toasts, dismissToast } = useToast();
    if (toasts.length === 0) return null;
    return (
        <div
            className="pointer-events-none fixed inset-x-0 bottom-2 z-50 flex flex-col items-end gap-1 px-2 sm:right-2 sm:bottom-2 sm:left-auto sm:items-end sm:px-0"
            aria-live="polite"
            aria-atomic="false"
        >
            {toasts.map((t) => {
                const s = STYLE[t.level];
                const Icon = s.icon;
                return (
                    <div
                        key={t.id}
                        className={cn(
                            "pointer-events-auto flex w-full max-w-sm items-start gap-2 rounded-md border border-border border-l-4 p-2 shadow-md animate-fade-in",
                            s.bg,
                            s.border,
                        )}
                        role={t.level === "error" || t.level === "warn" ? "alert" : "status"}
                    >
                        <Icon className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden />
                        <div className="min-w-0 flex-1 text-xs">
                            <p className="font-medium leading-tight">{t.title}</p>
                            {t.message && (
                                <p className="mt-0.5 text-muted-foreground">{t.message}</p>
                            )}
                        </div>
                        <button
                            type="button"
                            onClick={() => dismissToast(t.id)}
                            className="text-muted-foreground hover:text-foreground"
                            aria-label="Dismiss"
                        >
                            <X className="h-3.5 w-3.5" />
                        </button>
                    </div>
                );
            })}
        </div>
    );
}
