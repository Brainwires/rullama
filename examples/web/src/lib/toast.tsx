import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

export type ToastLevel = "info" | "success" | "warn" | "error";

export interface Toast {
    id:      string;
    level:   ToastLevel;
    title:   string;
    message?: string;
    persist?: boolean;   // if true, stays until user dismisses
    /** Optional inline action button. When clicked, `action.onClick`
     *  fires AND the toast is auto-dismissed. Used e.g. for the
     *  crash-detect "Open Logs" deep-link on next page load. */
    action?: {
        label:   string;
        onClick: () => void;
    };
}

interface ToastContextValue {
    toasts:       Toast[];
    showToast:    (t: Omit<Toast, "id"> & { id?: string }) => string;
    dismissToast: (id: string) => void;
    clearToasts:  () => void;
}

const ToastCtx = createContext<ToastContextValue | null>(null);

const AUTO_DISMISS_MS = 4500;

export function ToastProvider({ children }: { children: ReactNode }) {
    const [toasts, setToasts] = useState<Toast[]>([]);
    const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());

    const dismissToast = useCallback((id: string) => {
        const t = timers.current.get(id);
        if (t) { clearTimeout(t); timers.current.delete(id); }
        setToasts((cur) => cur.filter((x) => x.id !== id));
    }, []);

    const showToast = useCallback((t: Omit<Toast, "id"> & { id?: string }): string => {
        const id = t.id ?? `t_${Math.random().toString(36).slice(2, 9)}`;
        // Replace any existing toast with the same id so callers can update.
        setToasts((cur) => {
            const filtered = cur.filter((x) => x.id !== id);
            return [...filtered, { ...t, id }];
        });
        // Auto-dismiss for non-persistent, non-error/warn levels.
        const isSticky = t.persist || t.level === "error" || t.level === "warn";
        if (!isSticky) {
            const handle = setTimeout(() => dismissToast(id), AUTO_DISMISS_MS);
            const prev = timers.current.get(id);
            if (prev) clearTimeout(prev);
            timers.current.set(id, handle);
        }
        return id;
    }, [dismissToast]);

    const clearToasts = useCallback(() => {
        for (const h of timers.current.values()) clearTimeout(h);
        timers.current.clear();
        setToasts([]);
    }, []);

    const value = useMemo<ToastContextValue>(
        () => ({ toasts, showToast, dismissToast, clearToasts }),
        [toasts, showToast, dismissToast, clearToasts],
    );

    return <ToastCtx.Provider value={value}>{children}</ToastCtx.Provider>;
}

export function useToast(): ToastContextValue {
    const ctx = useContext(ToastCtx);
    if (!ctx) throw new Error("useToast must be used inside <ToastProvider>");
    return ctx;
}
