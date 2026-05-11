import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** ShadCN-style classname merge helper. */
export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

/** Bytes → human-readable string. */
export function fmtBytes(n: number | undefined | null): string {
    if (!n || !isFinite(n)) return "0 B";
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(0)} KB`;
    return `${n} B`;
}
