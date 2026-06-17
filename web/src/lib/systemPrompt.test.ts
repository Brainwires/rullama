import { describe, it, expect } from "vitest";
import { buildSysContent } from "./systemPrompt";
import { THINK_TOKEN, TIMESTAMP_SYSTEM_NOTE, FORMATTING_SYSTEM_NOTE } from "./app-helpers";
import { TOOL_SCHEMA_PROMPT } from "./toolFormat";

describe("buildSysContent", () => {
    it("appends the timestamp + formatting notes to a plain prompt", () => {
        const out = buildSysContent({ systemPrompt: "You are X.", thinking: false, toolMode: false });
        expect(out).toBe(`You are X.\n\n${TIMESTAMP_SYSTEM_NOTE}\n\n${FORMATTING_SYSTEM_NOTE}`);
    });

    it("wraps in the thinking control token when thinking is on", () => {
        const out = buildSysContent({ systemPrompt: "Hi", thinking: true, toolMode: false });
        expect(out.startsWith(`${THINK_TOKEN}\n`)).toBe(true);
    });

    it("prepends the tool schema in tool mode", () => {
        const out = buildSysContent({ systemPrompt: "Hi", thinking: false, toolMode: true });
        expect(out.startsWith(`${TOOL_SCHEMA_PROMPT}\n\n`)).toBe(true);
        expect(out).toContain("Hi");
    });

    it("handles an empty system prompt (notes still present)", () => {
        const out = buildSysContent({ systemPrompt: "", thinking: false, toolMode: false });
        expect(out).toBe(`${TIMESTAMP_SYSTEM_NOTE}\n\n${FORMATTING_SYSTEM_NOTE}`);
    });

    it("is byte-identical for the WARM path (no rag/gps) vs a plain turn", () => {
        // The whole pre-warm hot-start hinges on this: the warmed system
        // block must equal what a plain (no-RAG, no-GPS) turn renders, or
        // the KV cache wouldn't reuse it.
        for (const thinking of [false, true]) {
            for (const toolMode of [false, true]) {
                const warm = buildSysContent({ systemPrompt: "You are X.", thinking, toolMode });
                const plainTurn = buildSysContent({
                    systemPrompt: "You are X.", thinking, toolMode,
                    ragPreamble: "", gpsLine: "",
                });
                expect(warm).toBe(plainTurn);
            }
        }
    });

    it("injects rag preamble and gps line at their layered positions", () => {
        const out = buildSysContent({
            systemPrompt: "SYS", thinking: false, toolMode: true,
            ragPreamble: "RAG\n\n", gpsLine: "GPS\n\n",
        });
        // GPS leads (tool-mode front), then tool schema, then rag, then sys.
        expect(out.startsWith("GPS\n\n")).toBe(true);
        expect(out.indexOf("RAG\n\n")).toBeGreaterThan(out.indexOf(TOOL_SCHEMA_PROMPT));
        expect(out.indexOf("SYS")).toBeGreaterThan(out.indexOf("RAG\n\n"));
    });
});
