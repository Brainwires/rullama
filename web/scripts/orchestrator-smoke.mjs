// End-to-end smoke for the vendored tool-orchestrator WASM + the memoized-replay
// async bridge that lib/tools/orchestrator.ts implements. Loads the REAL wasm,
// registers async-mocked tools, and runs scripts that mirror what Gemma writes
// (single / dependent / fan-out loop / numeric branch / warmest-of-N) — checking
// object access, numeric comparison, and replay convergence.
//
//   node web/scripts/orchestrator-smoke.mjs
//
// Pure-JS, no app imports (jsdom can't load a /public wasm URL, so this lives
// outside vitest). The pure parts of orchestrator.ts are unit-tested in
// orchestrator.test.ts.
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const { default: init, WasmOrchestrator, ExecutionLimits } = await import(
    here("../public/tool-orchestrator/tool_orchestrator.js")
);
await init(await readFile(here("../public/tool-orchestrator/tool_orchestrator_bg.wasm")));

const TEMPS = { Tokyo: 18, Paris: 24, Miami: 31 };
async function tool(name, args) {
    await new Promise((r) => setTimeout(r, 5)); // force async, like fetch()
    if (name === "get_weather") {
        const c = args.location;
        return { ok: true, summary: `${c} is ${TEMPS[c] ?? 20}C, Cloudy`, temp_c: TEMPS[c] ?? 20 };
    }
    if (name === "get_air_quality") return { ok: true, summary: `Air in ${args.location}: AQI 42 (Good)`, aqi: 42 };
    return { ok: false, summary: `no tool ${name}` };
}

const TOOL_PARAMS = { get_weather: ["location"], get_air_quality: ["location"] };
const NAMES = Object.keys(TOOL_PARAMS);

function normArgs(name, jsonArgs) {
    let v;
    try { v = JSON.parse(jsonArgs); } catch { v = jsonArgs; }
    if (v && typeof v === "object" && !Array.isArray(v)) return v;
    return { [TOOL_PARAMS[name][0]]: v };
}

async function run(script) {
    const cache = new Map();
    for (let pass = 1; pass <= 12; pass++) {
        const misses = new Map();
        const orch = new WasmOrchestrator();
        for (const name of NAMES) {
            orch.register_tool(name, (jsonArgs) => {
                const key = `${name} ${jsonArgs}`;
                if (cache.has(key)) return cache.get(key);
                if (!misses.has(key)) misses.set(key, { name, args: normArgs(name, jsonArgs) });
                throw new Error("__PENDING__");
            });
        }
        const result = orch.execute(script, ExecutionLimits.extended());
        if (misses.size === 0) return { ok: result.success, output: result.output, error: result.error, passes: pass };
        await Promise.all([...misses.entries()].map(async ([key, { name, args }]) => {
            const r = await tool(name, args);
            cache.set(key, JSON.stringify({ ok: r.ok, summary: r.summary, ...r }));
        }));
    }
    return { ok: false, error: "no converge", passes: 12 };
}

const cases = [
    ["single", `let w = get_weather("Tokyo");\nw.summary`, (o) => o.includes("Tokyo is 18C")],
    ["dependent", `let w = get_weather("Tokyo");\nlet aq = get_air_quality("Tokyo");\n"W: " + w.summary + " | A: " + aq.summary`, (o) => o.includes("Tokyo is 18C") && o.includes("AQI 42")],
    ["fanout-loop", `let cities = ["Tokyo","Paris","Miami"];\nlet out = "";\nfor c in cities { let w = get_weather(c); out += c + ": " + w.summary + "\\n"; }\nout`, (o) => o.includes("Tokyo is 18C") && o.includes("Paris is 24C") && o.includes("Miami is 31C")],
    ["numeric-branch", `let w = get_weather("Miami");\nif w.temp_c > 20 { "warm: " + w.temp_c } else { "cool" }`, (o) => o.includes("warm: 31")],
    ["warmest", `let best = ""; let max = -999;\nfor c in ["Tokyo","Paris","Miami"] { let w = get_weather(c); if w.temp_c > max { max = w.temp_c; best = c; } }\n"Warmest: " + best`, (o) => o.includes("Warmest: Miami")],
];

let fail = 0;
for (const [label, script, check] of cases) {
    const r = await run(script);
    const ok = r.ok && check(r.output ?? "");
    if (!ok) fail++;
    console.log(`${ok ? "PASS" : "FAIL"}  ${label.padEnd(16)} passes=${r.passes}  out=${JSON.stringify(r.output ?? r.error)}`);
}
console.log(fail === 0 ? "\nALL PASS ✓" : `\n${fail} FAILED ✗`);
process.exit(fail === 0 ? 0 : 1);
