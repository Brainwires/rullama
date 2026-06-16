import { describe, it, expect } from "vitest";
import { parseToolCalls } from "@/lib/parseToolCalls";

const wrap = (inner: string) => `<tool_call>${inner}</tool_call>`;

describe("parseToolCalls — JSON form", () => {
    it("passes through prose with no tool call", () => {
        const r = parseToolCalls("I can't do that directly.");
        expect(r.calls).toEqual([]);
        expect(r.prose).toBe("I can't do that directly.");
        expect(r.pending).toBe(false);
    });

    it("parses a single clean JSON call", () => {
        const r = parseToolCalls(wrap('{"name":"get_weather","arguments":{"location":"Miami"}}'));
        expect(r.calls).toHaveLength(1);
        expect(r.calls[0]).toMatchObject({ name: "get_weather", arguments: { location: "Miami" }, pending: false });
        expect(r.prose).toBe("");
    });

    it("maps the `parameters` alias onto arguments", () => {
        const r = parseToolCalls(wrap('{"name":"set_timer","parameters":{"duration":5}}'));
        expect(r.calls[0]).toMatchObject({ name: "set_timer", arguments: { duration: 5 } });
    });

    it("extracts prose around a call", () => {
        const r = parseToolCalls(`Sure! ${wrap('{"name":"play_music","arguments":{"query":"jazz"}}')} done.`);
        expect(r.calls).toHaveLength(1);
        expect(r.prose).toContain("Sure!");
        expect(r.prose).toContain("done.");
    });

    it("parses multiple JSON calls", () => {
        const r = parseToolCalls(
            wrap('{"name":"set_timer","arguments":{"duration":1}}') +
            wrap('{"name":"set_timer","arguments":{"duration":2}}'),
        );
        expect(r.calls).toHaveLength(2);
        expect(r.calls[1].arguments).toEqual({ duration: 2 });
    });
});

describe("parseToolCalls — pythonic form (small-model fallback)", () => {
    it("maps positional args to schema param names", () => {
        const r = parseToolCalls(wrap("set_timer(7)"));
        expect(r.calls[0]).toMatchObject({ name: "set_timer", arguments: { duration: 7 } });
    });

    it("parses keyword args", () => {
        const r = parseToolCalls(wrap('send_email(to="Priya", subject="Budget Review")'));
        expect(r.calls[0].arguments).toEqual({ to: "Priya", subject: "Budget Review" });
    });

    it("maps multiple positional args (with commas inside quoted strings)", () => {
        const r = parseToolCalls(wrap('set_reminder("call grandma, tonight", "tonight")'));
        expect(r.calls[0]).toMatchObject({
            name: "set_reminder",
            arguments: { text: "call grandma, tonight", time: "tonight" },
        });
    });

    it("coerces numeric and bareword scalars", () => {
        const r = parseToolCalls(wrap("get_weather(Tokyo)"));
        expect(r.calls[0].arguments).toEqual({ location: "Tokyo" });
    });

    it("falls back to arg0.. for unknown tools with positional args", () => {
        const r = parseToolCalls(wrap('unknown_tool("x")'));
        expect(r.calls[0]).toMatchObject({ name: "unknown_tool", arguments: { arg0: "x" } });
    });
});

describe("parseToolCalls — robustness (real model quirks)", () => {
    it("tolerates a dropped '>' on the open tag", () => {
        const r = parseToolCalls('<tool_call\n{"name":"set_timer","arguments":{"timer_duration":7}}</tool_call>');
        expect(r.calls[0]).toMatchObject({ name: "set_timer" });
    });

    it("brace-matches a missing </tool_call> closer + trailing junk", () => {
        const r = parseToolCalls('<tool_call>{"name":"audio","arguments":{}}}}');
        expect(r.calls).toHaveLength(1);
        expect(r.calls[0]).toMatchObject({ name: "audio", pending: false });
    });

    it("marks a still-streaming mid-JSON call pending", () => {
        const r = parseToolCalls('<tool_call>{"name":"get_w');
        expect(r.calls[0].pending).toBe(true);
        expect(r.pending).toBe(true);
    });

    it("keeps genuinely malformed payloads raw without throwing", () => {
        const r = parseToolCalls(wrap("not json and not a call !!!"));
        expect(r.calls).toHaveLength(1);
        expect(typeof r.calls[0].arguments).toBe("string");
    });
});

describe("parseToolCalls — regression: actual base-model eval outputs", () => {
    const cases: Array<[string, string, Record<string, unknown>]> = [
        ['{"name":"get_weather","arguments":{"location":"Miami"}}', "get_weather", { location: "Miami" }],
        ['{"name":"play_music","arguments":{"query":"classical music"}}', "play_music", { query: "classical music" }],
        ["set_timer(7)", "set_timer", { duration: 7 }],
        ['send_email(to="Priya", subject="Budget Review")', "send_email", { to: "Priya", subject: "Budget Review" }],
        ['set_reminder("call grandma tonight", "tonight")', "set_reminder", { text: "call grandma tonight", time: "tonight" }],
    ];
    it.each(cases)("parses %s", (inner, name, args) => {
        const r = parseToolCalls(wrap(inner));
        expect(r.calls).toHaveLength(1);
        expect(r.calls[0].name).toBe(name);
        expect(r.calls[0].arguments).toEqual(args);
    });
});
