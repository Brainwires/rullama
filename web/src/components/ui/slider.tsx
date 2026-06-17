import { cn } from "@/lib/utils";

interface Props {
    label:   string;
    value:   number;
    min:     number;
    max:     number;
    step:    number;
    onChange:(v: number) => void;
    /** Optional fixed-decimals formatter for the displayed value. */
    fmt?:   (v: number) => string;
    className?: string;
}

/**
 * Compact labeled slider — label + value above, native range input below.
 * Native `<input type="range">` is keyboard-accessible out of the box and
 * needs no extra dependency. The track + thumb are restyled via the
 * `accent-*` tailwind utility (works in modern Chromium, Safari, Firefox).
 */
export function Slider({ label, value, min, max, step, onChange, fmt, className }: Props) {
    const display = fmt ? fmt(value) : (Number.isInteger(step) ? String(value) : value.toFixed(2));
    return (
        <label className={cn("flex flex-col gap-0.5 text-[0.65rem] uppercase tracking-wider text-muted-foreground", className)}>
            <span className="flex items-baseline justify-between gap-2">
                <span>{label}</span>
                <span className="font-mono text-[0.7rem] normal-case tracking-normal text-foreground">{display}</span>
            </span>
            <input
                type="range"
                value={value}
                min={min}
                max={max}
                step={step}
                onChange={(e) => onChange(Number(e.target.value))}
                className="h-5 w-full cursor-pointer appearance-none rounded-md bg-muted accent-primary"
            />
        </label>
    );
}
