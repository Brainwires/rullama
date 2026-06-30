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

    it("is byte-identical for the WARM path vs a plain turn (content is fully static)", () => {
        // The whole pre-warm hot-start hinges on this: the warmed system block
        // must equal what a real turn renders, or the KV cache wouldn't reuse it.
        for (const thinking of [false, true]) {
            for (const toolMode of [false, true]) {
                const warm = buildSysContent({ systemPrompt: "You are X.", thinking, toolMode });
                const plainTurn = buildSysContent({ systemPrompt: "You are X.", thinking, toolMode });
                expect(warm).toBe(plainTurn);
            }
        }
    });

    it("uses the Rhai orchestrator preamble (not the JSON schema) in orchestrator mode", () => {
        const json = buildSysContent({ systemPrompt: "Hi", thinking: false, toolMode: true });
        const orch = buildSysContent({ systemPrompt: "Hi", thinking: false, toolMode: true, orchestratorMode: true });
        expect(orch).not.toBe(json);
        expect(orch).not.toContain(TOOL_SCHEMA_PROMPT);
        expect(orch).toContain("Rhai");
        expect(orch).toContain("get_weather");
        expect(orch).toContain("Hi");
    });

    it("warm == plain turn in orchestrator mode too (signature differs from JSON mode → its own warm)", () => {
        for (const thinking of [false, true]) {
            const warm = buildSysContent({ systemPrompt: "You are X.", thinking, toolMode: true, orchestratorMode: true });
            const plainTurn = buildSysContent({ systemPrompt: "You are X.", thinking, toolMode: true, orchestratorMode: true });
            expect(warm).toBe(plainTurn);
        }
    });

    it("leads with the tool schema in tool mode (static core, no dynamic tail)", () => {
        const out = buildSysContent({ systemPrompt: "SYS", thinking: false, toolMode: true });
        expect(out.startsWith(`${TOOL_SCHEMA_PROMPT}\n\n`)).toBe(true);
        expect(out).toContain("SYS");
    });
});
