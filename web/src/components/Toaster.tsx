import { useToast, type ToastLevel } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { X, AlertTriangle, CircleAlert, Info, CheckCircle2 } from "lucide-react";

// Opaque card background (readable over any page content) + a level-colored
// left border and icon. The old translucent bg-*/10 fills let the page show
// through and were hard to read.
const STYLE: Record<ToastLevel, { border: string; iconColor: string; icon: typeof Info }> = {
    error:   { border: "border-l-destructive",      iconColor: "text-destructive",      icon: CircleAlert },
    warn:    { border: "border-l-yellow-500",       iconColor: "text-yellow-500",       icon: AlertTriangle },
    success: { border: "border-l-emerald-500",      iconColor: "text-emerald-500",      icon: CheckCircle2 },
    info:    { border: "border-l-muted-foreground", iconColor: "text-muted-foreground", icon: Info },
};

/** Bottom-right toast stack. Persistent (error/warn) toasts stay until
 *  dismissed; info/success auto-dismiss in the provider. */
export function Toaster() {
    const { toasts, dismissToast } = useToast();
    if (toasts.length === 0) return null;
    return (
        <div
            className="pointer-events-none fixed inset-x-0 z-50 flex flex-col items-center gap-1 px-2"
            // Top-anchored, with a thick top offset that clears the header
            // (min-h-12 ≈ 3rem) plus the iOS safe-area inset. flex-col stacks
            // the newest toast at the top of the column.
            style={{ top: `calc(env(safe-area-inset-top, 0px) + 4.5rem)` }}
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
                            "pointer-events-auto flex w-full max-w-sm items-start gap-2 rounded-md border border-border border-l-4 bg-card p-2 text-card-foreground shadow-lg animate-fade-in",
                            s.border,
                        )}
                        role={t.level === "error" || t.level === "warn" ? "alert" : "status"}
                    >
                        <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", s.iconColor)} aria-hidden />
                        <div className="min-w-0 flex-1 text-xs">
                            <p className="font-medium leading-tight">{t.title}</p>
                            {t.message && (
                                <p className="mt-0.5 text-muted-foreground">{t.message}</p>
                            )}
                            {t.action && (
                                <button
                                    type="button"
                                    onClick={() => { try { t.action!.onClick(); } finally { dismissToast(t.id); } }}
                                    className="mt-1 rounded border border-border bg-background px-2 py-0.5 text-[11px] font-medium text-foreground hover:bg-accent"
                                >
                                    {t.action.label}
                                </button>
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
