// Programmatic Tool Calling via Brainwires/tool-orchestrator (Rhai → WASM).
//
// Instead of the sequential JSON `<tool_call>` loop (one model round-trip per
// tool), the model writes ONE Rhai script that orchestrates many tools; only
// the final result returns. 37–98% fewer tokens on multi-tool turns. This is
// the OPT-IN "orchestrator mode" — it coexists with, and falls back to, the
// JSON loop (useChatEngine compiles+runs the script; any failure drops to the
// loop). See the v-0.6 plan, Phase B.
//
// THE ASYNC BRIDGE (load-bearing): the WASM engine runs scripts SYNCHRONOUSLY —
// it calls each registered tool as a sync JS fn that must return a string NOW.
// Our tools are async (`fetch`). We bridge with MEMOIZED REPLAY: run the script;
// every not-yet-resolved tool call records a "miss" and throws (the engine turns
// that into an error value, so the script aborts on first use of it); resolve all
// misses concurrently; re-run with those results cached. Deterministic tools
// converge — each pass resolves ≥1 new call, so a script of tool-depth N settles
// in ≤N passes. Crucially this is CORRECT for data-dependent control flow: by the
// time a branch is re-evaluated, the value it branches on is the REAL cached
// result, not a placeholder.

import { executeTool } from "@/lib/tools";
import { TOOL_PARAMS } from "@/lib/toolFormat";
import type { ToolContext } from "@/lib/tools/types";

// ─── WASM module load (lazy; mirrors compute.ts) ──────────────────────────

interface OrchestratorModule {
    default: (wasm: string) => Promise<unknown>;
    WasmOrchestrator: new () => {
        register_tool(name: string, cb: (jsonArgs: string) => string): void;
        execute(script: string, limits: unknown): unknown;
    };
    ExecutionLimits: { extended(): unknown };
}

let modPromise: Promise<OrchestratorModule> | null = null;
function loadOrchestrator(): Promise<OrchestratorModule> {
    if (!modPromise) {
        modPromise = (async () => {
            // Variable specifier (not a literal) so tsc/Vite don't try to resolve
            // the /public runtime URL as a bundled module.
            const jsUrl = "/tool-orchestrator/tool_orchestrator.js";
            const mod = (await import(/* @vite-ignore */ jsUrl)) as OrchestratorModule;
            await mod.default("/tool-orchestrator/tool_orchestrator_bg.wasm");
            return mod;
        })();
    }
    return modPromise;
}

/** Tools exposed to the orchestrator: every executable tool EXCEPT the math
 *  `calculate` gateway (its progressive-disclosure unlock is a JSON-loop
 *  construct — meaningless inside a self-contained script). Names are the
 *  single source of truth (TOOL_PARAMS), so the script's tool surface tracks
 *  the JSON loop's automatically. */
const ORCH_TOOL_NAMES: string[] = Object.keys(TOOL_PARAMS).filter((n) => n !== "calculate");

// ─── The orchestrator preamble (syntax-teaching + few-shot) ───────────────
//
// The B0 spike proved e2b plans orchestration correctly but defaults to LUA
// syntax (`then…end`, `..`, `and`). These explicit Rhai rules + worked examples
// flipped it to mostly-valid Rhai (Task 1 + Task 3 perfect, Task 2 near). Keep
// this in lock-step with what the engine actually accepts.

function signatureLines(): string {
    return ORCH_TOOL_NAMES.map((n) => `  ${n}(${TOOL_PARAMS[n].join(", ")})`).join("\n");
}

export function orchestratorPreamble(): string {
    return `You orchestrate tools by writing ONE script in Rhai (a Rust-like scripting language). \
The script runs locally; only its final result is shown to the user, so do all the work in the script.

Available tools — each performs an action and returns an object with a \`.summary\` string (the human-readable result):
${signatureLines()}

Rhai syntax — Rhai is NOT Lua or Python:
  - Blocks use BRACES, never then/end:        if x > 20 { ... } else { ... }
  - String concatenation uses + :             "a" + "b"                 (NOT "..")
  - Logical and / or are && and || :          a > 0 && b > 0            (NOT and/or)
  - End every statement with a semicolon ;
  - Use ONLY + to build strings — there is no string.format / printf.
  - The script's FINAL expression is the returned result; make it a STRING.

Examples:

  // one tool
  let w = get_weather("Tokyo");
  w.summary

  // fan out, then combine (this is the big win — one script, no round-trips)
  let cities = ["Tokyo", "Paris", "Miami"];
  let out = "";
  for c in cities {
      let w = get_weather(c);
      out += c + ": " + w.summary + "\\n";
  }
  out

  // dependent step
  let w = get_weather("Tokyo");
  let aq = get_air_quality("Tokyo");
  "Weather: " + w.summary + " Air: " + aq.summary

Reply with ONLY a Rhai script — no prose, no markdown fences. If no tool is needed, reply with a normal answer instead of a script.`;
}

// ─── Script extraction ────────────────────────────────────────────────────

/** Pull a Rhai script out of a model reply: strip ``` fences if present, else
 *  take the text verbatim. Returns null for an empty/whitespace body. */
export function extractScript(reply: string): string | null {
    let s = reply.trim();
    const fence = s.match(/```(?:rhai|rust|js)?\s*\n([\s\S]*?)```/i);
    if (fence) s = fence[1].trim();
    return s.length > 0 ? s : null;
}

// ─── Result types ─────────────────────────────────────────────────────────

export interface OrchestrationCall {
    name: string;
    args: Record<string, unknown>;
    summary: string;
    ok: boolean;
}

export interface OrchestrationResult {
    /** True if the script compiled, executed, and produced output. */
    ok: boolean;
    /** The script's final value (the user-facing answer) when ok. */
    output: string;
    /** Failure reason (compile/exec error, no-converge) when !ok — the signal
     *  to fall back to the JSON loop. */
    error: string | null;
    /** Tools actually run (deduped), for optional surfacing. */
    calls: OrchestrationCall[];
    /** Replay passes it took to settle (diagnostics). */
    passes: number;
}

const MAX_REPLAYS = 12;
const PENDING = "__RULLAMA_PENDING__";

function getField(obj: unknown, key: string): unknown {
    if (obj == null) return undefined;
    if (obj instanceof Map) return obj.get(key);
    return (obj as Record<string, unknown>)[key];
}

function safeParseArgs(jsonArgs: string): Record<string, unknown> {
    try {
        const v = JSON.parse(jsonArgs);
        if (v && typeof v === "object" && !Array.isArray(v)) return v as Record<string, unknown>;
        // Rhai passes a bare scalar (e.g. a string arg) — wrap onto the first
        // declared param so tools see a named key.
        return { __scalar__: v };
    } catch {
        return { __scalar__: jsonArgs };
    }
}

/** Map a bare-scalar Rhai arg onto the tool's first declared parameter name,
 *  so `get_weather("Tokyo")` reaches the weather tool as `{ location: "Tokyo" }`. */
function normalizeArgs(name: string, args: Record<string, unknown>): Record<string, unknown> {
    if ("__scalar__" in args) {
        const params = TOOL_PARAMS[name];
        const key = params && params.length > 0 ? params[0] : "input";
        return { [key]: args.__scalar__ };
    }
    return args;
}

/**
 * Compile + run a model-written Rhai script against the live tool registry,
 * bridging async tools via memoized replay. Never throws — failures come back
 * as `{ ok: false, error }` so the caller can fall back to the JSON loop.
 */
export async function runOrchestration(
    script: string,
    ctx: ToolContext,
): Promise<OrchestrationResult> {
    let mod: OrchestratorModule;
    try {
        mod = await loadOrchestrator();
    } catch (e) {
        return { ok: false, output: "", error: `orchestrator load failed: ${(e as Error).message}`, calls: [], passes: 0 };
    }

    const cache = new Map<string, string>(); // callKey → JSON result string
    const calls = new Map<string, OrchestrationCall>(); // callKey → record

    for (let pass = 1; pass <= MAX_REPLAYS; pass++) {
        const misses = new Map<string, { name: string; args: Record<string, unknown> }>();

        const orch = new mod.WasmOrchestrator();
        for (const name of ORCH_TOOL_NAMES) {
            orch.register_tool(name, (jsonArgs: string): string => {
                const key = `${name} ${jsonArgs}`;
                const hit = cache.get(key);
                if (hit !== undefined) return hit;
                if (!misses.has(key)) {
                    misses.set(key, { name, args: normalizeArgs(name, safeParseArgs(jsonArgs)) });
                }
                // Throw so the engine records an error value for this call; the
                // script then aborts on first use of it. We only consume output
                // from a pass with ZERO misses, so this pass's result is ignored.
                throw new Error(`${PENDING}:${name}`);
            });
        }

        let result: unknown;
        try {
            result = orch.execute(script, mod.ExecutionLimits.extended());
        } catch (e) {
            return { ok: false, output: "", error: `engine error: ${(e as Error).message}`, calls: [...calls.values()], passes: pass };
        }

        if (misses.size === 0) {
            const success = Boolean(getField(result, "success"));
            const output = String(getField(result, "output") ?? "");
            const err = getField(result, "error");
            if (!success) {
                return { ok: false, output: "", error: typeof err === "string" ? err : "script execution failed", calls: [...calls.values()], passes: pass };
            }
            return { ok: true, output, error: null, calls: [...calls.values()], passes: pass };
        }

        // Resolve every miss concurrently, cache the real (deterministic) result.
        await Promise.all(
            [...misses.entries()].map(async ([key, { name, args }]) => {
                let summary: string;
                let ok: boolean;
                let data: Record<string, unknown> | undefined;
                try {
                    const r = await executeTool(name, args, ctx);
                    ok = r.ok;
                    summary = r.summary;
                    data = r.data;
                } catch (e) {
                    ok = false;
                    summary = `tool error: ${(e as Error).message}`;
                }
                cache.set(key, JSON.stringify({ ok, summary, ...(data ?? {}) }));
                calls.set(key, { name, args, summary, ok });
            }),
        );
    }

    return { ok: false, output: "", error: `did not converge in ${MAX_REPLAYS} passes`, calls: [...calls.values()], passes: MAX_REPLAYS };
}
