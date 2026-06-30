import { describe, it, expect } from "vitest";
import { formatTurnTimestamp, withTurnTimestamp } from "./app-helpers";

describe("per-turn timestamp injection", () => {
    it("formats epoch-ms as a stable 'Wkd YYYY-MM-DD HH:MM TZ' string", () => {
        const ms = 1_700_000_000_000;
        // weekday, date, time, and (when resolvable) a timezone abbreviation.
        expect(formatTurnTimestamp(ms)).toMatch(/^[A-Za-z]{3} \d{4}-\d{2}-\d{2} \d{2}:\d{2}(?: \S+)?$/);
    });

    it("is deterministic — same ms always yields the same string", () => {
        // The invariant KV-cache reuse depends on: re-rendering a frozen
        // timestamp must never drift.
        const ms = 1_705_318_920_000;
        expect(formatTurnTimestamp(ms)).toBe(formatTurnTimestamp(ms));
    });

    it("prepends the bracketed stamp to user content", () => {
        const ms = 1_700_000_000_000;
        expect(withTurnTimestamp("hello", ms)).toBe(`[${formatTurnTimestamp(ms)}] hello`);
    });

    it("passes content through unchanged when no timestamp is known", () => {
        // Legacy / resume turns with no createdAt must NOT get a stamp —
        // an unreproducible "now" would poison the cached prefix.
        expect(withTurnTimestamp("hello", undefined)).toBe("hello");
    });
});
