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

/** Clamp a number to [min, max]. NaN, ±Infinity, and non-numeric inputs
 *  fall back to `fallback` (so a blank or pasted-garbage input becomes a
 *  predictable value rather than silently breaking sampling). */
export function clampNum(
    value: unknown,
    min: number,
    max: number,
    fallback: number,
): number {
    const n = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(n)) return fallback;
    if (n < min) return min;
    if (n > max) return max;
    return n;
}

/** Clamp an integer; same fallbacks as `clampNum`. */
export function clampInt(
    value: unknown,
    min: number,
    max: number,
    fallback: number,
): number {
    const n = Math.trunc(clampNum(value, min, max, fallback));
    return n;
}
