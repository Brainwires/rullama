import { describe, it, expect } from "vitest";
import { extractScript, orchestratorPreamble } from "./orchestrator";
import { TOOL_PARAMS } from "@/lib/toolFormat";

// The WASM-backed runOrchestration is covered end-to-end by the node smoke
// (web/scripts/orchestrator-smoke.mjs); here we lock the pure parts.

describe("orchestratorPreamble", () => {
    it("lists every executable tool except the calculate gateway, with signatures", () => {
        const p = orchestratorPreamble();
        for (const name of Object.keys(TOOL_PARAMS)) {
            if (name === "calculate") {
                expect(p).not.toContain("calculate(");
            } else {
                expect(p).toContain(`${name}(${TOOL_PARAMS[name].join(", ")})`);
            }
        }
    });

    it("teaches Rhai syntax (braces, +, &&) — the spike's residual gap", () => {
        const p = orchestratorPreamble();
        expect(p).toContain("Rhai");
        expect(p).toMatch(/NOT Lua/i);
        expect(p).toContain("&&");
        expect(p).toContain(".summary");
    });
});

describe("extractScript", () => {
    it("returns a bare script verbatim", () => {
        expect(extractScript('let w = get_weather("Tokyo");\nw.summary')).toBe('let w = get_weather("Tokyo");\nw.summary');
    });
    it("strips ``` fences (with or without a language tag)", () => {
        expect(extractScript('```rhai\nget_weather("Tokyo")\n```')).toBe('get_weather("Tokyo")');
        expect(extractScript('```\nlet x = 1; x\n```')).toBe('let x = 1; x');
    });
    it("returns null for empty / whitespace", () => {
        expect(extractScript("   \n  ")).toBeNull();
        expect(extractScript("")).toBeNull();
    });
});
