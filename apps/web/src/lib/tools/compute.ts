// compute-engine integration via a `calculate` GATEWAY (progressive disclosure).
//
// The top-level schema advertises ONE tool, `calculate("<problem>")`. Calling it
// doesn't compute — it UNLOCKS the math verbs (their tool_response is the
// sub-schema), and the model then makes the precise second call. The verbs are
// registered (executable) but NOT listed top-level, so math costs 1 schema slot.
//
// Backed by Brainwires/compute-engine (Rust → WASM, 400+ ops), lazy-loaded from
// /public on first use so its 3.5 MB never touches the initial bundle. Verbs are
// thin pass-throughs: the model's args go straight to engine.<verb>(args) and
// the engine validates — so we don't have to perfectly model its input schema.

import type { ToolDef, ToolRunResult } from "@/lib/tools/types";

// Lazy singleton — the WASM (web target) is served as a static asset; @vite-ignore
// keeps Vite from trying to bundle the runtime URL.
let enginePromise: Promise<{ solve(i: unknown): unknown; analyze(i: unknown): unknown; units(i: unknown): unknown }> | null = null;
function getEngine() {
    if (!enginePromise) {
        enginePromise = (async () => {
            // Variable specifier (not a literal) so tsc doesn't try to resolve
            // the runtime /public URL as a module.
            const jsUrl = "/compute-engine/computational_engine.js";
            const mod = await import(/* @vite-ignore */ jsUrl) as {
                default: (wasm: string) => Promise<unknown>;
                ComputationalEngine: new () => {
                    solve(i: unknown): unknown; analyze(i: unknown): unknown; units(i: unknown): unknown;
                };
            };
            await mod.default("/compute-engine/computational_engine_bg.wasm");
            return new mod.ComputationalEngine();
        })();
    }
    return enginePromise;
}

/** Compact, model-friendly view of an engine result object. */
function summarize(r: unknown): string {
    if (r == null) return "(no result)";
    if (typeof r === "string") return r;
    const o = r as Record<string, unknown>;
    const pick = o.result ?? o.solutions ?? o.value ?? r;
    return typeof pick === "string" ? pick : JSON.stringify(pick);
}

async function runVerb(verb: "solve" | "analyze" | "units", args: Record<string, unknown>): Promise<ToolRunResult> {
    try {
        const engine = await getEngine();
        const result = engine[verb](args);
        const o = result as Record<string, unknown> | null;
        if (o && (o.error || o.success === false)) {
            return { ok: false, summary: `${verb} failed: ${o.error ?? "unknown error"}` };
        }
        return { ok: true, summary: summarize(result), data: { verb } };
    } catch (e) {
        return { ok: false, summary: `${verb} failed: ${(e as Error).message}. Check the argument shape.` };
    }
}

// ─── The gateway: 1 top-level tool that unlocks the math verbs ────────────

const UNLOCK =
    "Math engine ready. To answer the problem, call ONE of these tools (copy the JSON shape exactly):\n" +
    '- solve — equations & roots: solve({"equations":["x^2 - 4 = 0"]})\n' +
    '- analyze — simplify / expand / factor an expression: analyze({"operation":"Simplify","expression":"(x+1)^2"})  (operation is one of: Simplify, Expand, Factor)\n' +
    '- units — unit conversion: units({"operation":"Convert","value":5,"from_unit":"mile","to_unit":"km"})\n' +
    "Pick the one tool that fits, then call it.";

export const calculateTool: ToolDef = {
    names: ["calculate", "math", "compute_math"],
    run(_name, _args): Promise<ToolRunResult> {
        // The gateway never computes — it returns the sub-schema so the model
        // makes the precise second call (progressive disclosure).
        return Promise.resolve({ ok: true, summary: UNLOCK, data: { gateway: true } });
    },
};

// Unlocked verbs — registered (executable) but NOT in the top-level schema.
export const solveTool: ToolDef = {
    names: ["solve"],
    run: (_n, args) => runVerb("solve", args),
};
export const analyzeTool: ToolDef = {
    names: ["analyze"],
    run: (_n, args) => runVerb("analyze", args),
};
export const unitsTool: ToolDef = {
    names: ["units", "convert_units"],
    run: (_n, args) => runVerb("units", args),
};
