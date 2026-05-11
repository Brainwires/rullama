import { useCallback, useEffect, useRef, useState } from "react";

/**
 * `useState` that mirrors its value to `localStorage`, namespaced under
 * `rullama:<key>`. The initial render uses the persisted value when one
 * exists; otherwise the provided default. Writes are JSON-serialized.
 *
 * Returns `[value, setValue]` with the same shape as `useState`. The
 * setter accepts either a new value or an updater function — the new
 * value is written to localStorage synchronously inside `setValue`, so
 * a navigation-away right after a setState still keeps the change.
 *
 * Failures (quota exceeded, JSON.parse error, localStorage unavailable
 * in private mode) degrade silently — the hook behaves like a plain
 * useState in those cases.
 */
export function usePersistedState<T>(
    key: string,
    defaultValue: T,
): [T, (next: T | ((prev: T) => T)) => void] {
    const storageKey = `rullama:${key}`;
    const lastWrittenRef = useRef<string | null>(null);

    const [value, setValueInner] = useState<T>(() => {
        try {
            if (typeof window === "undefined") return defaultValue;
            const raw = window.localStorage.getItem(storageKey);
            if (raw == null) return defaultValue;
            lastWrittenRef.current = raw;
            return JSON.parse(raw) as T;
        } catch {
            return defaultValue;
        }
    });

    const setValue = useCallback(
        (next: T | ((prev: T) => T)) => {
            setValueInner((prev) => {
                const resolved = typeof next === "function"
                    ? (next as (p: T) => T)(prev)
                    : next;
                try {
                    if (typeof window !== "undefined") {
                        const serialized = JSON.stringify(resolved);
                        if (serialized !== lastWrittenRef.current) {
                            window.localStorage.setItem(storageKey, serialized);
                            lastWrittenRef.current = serialized;
                        }
                    }
                } catch { /* full quota / private mode — drop */ }
                return resolved;
            });
        },
        [storageKey],
    );

    // Pick up cross-tab changes to the same key.
    useEffect(() => {
        if (typeof window === "undefined") return;
        const onStorage = (ev: StorageEvent) => {
            if (ev.key !== storageKey) return;
            if (ev.newValue == null) return;
            try {
                const parsed = JSON.parse(ev.newValue) as T;
                lastWrittenRef.current = ev.newValue;
                setValueInner(parsed);
            } catch { /* */ }
        };
        window.addEventListener("storage", onStorage);
        return () => window.removeEventListener("storage", onStorage);
    }, [storageKey]);

    return [value, setValue];
}
