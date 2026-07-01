// One-shot GPU performance burst + cached result.
//
// The static capability probe (lib/capability.ts) reads `adapter.limits` /
// `deviceMemory`, which tell us *support* but not actual *throughput* — and
// the browser can't read VRAM or clocks. So we run a short, representative
// compute burst (a few naive matmuls — the same op the engine is bound on)
// ONCE, time it wall-clock, and classify the GPU into a performance class.
//
// **Run once, cache forever.** The result is persisted in localStorage keyed
// by a bench-version + a coarse adapter fingerprint; subsequent startups read
// the cached number and never re-run (the burst only happens on first visit,
// or after a bench-logic bump / GPU change). This keeps boot instant.
//
// Timing is wall-clock around `queue.onSubmittedWorkDone()` (not timestamp
// queries — those need a feature iOS Safari doesn't expose), after a warmup
// submit so pipeline creation isn't counted.

const CACHE_KEY = "rullama:gpuBench";
// Bump when the workload below changes (invalidates old cached numbers).
const BENCH_VERSION = 1;

// Workload size. M=N=K matmul, repeated REPS times in one submit. Sized so a
// fast desktop GPU finishes in a few ms and a weak mobile GPU in tens of ms —
// measurable on both without stalling anything. ~134 MFLOP/matmul × REPS.
const DIM = 512;
const REPS = 24;

export type PerfClass = "slow" | "mid" | "fast";

export interface BenchResult {
    ms:        number;     // wall-clock for REPS matmuls (lower = faster)
    perfClass: PerfClass;
    /** True when freshly measured this session (false = served from cache). */
    fresh:     boolean;
}

// Provisional breakpoints — TODO: CALIBRATE against real devices (iPhone 16e,
// a 12GB desktop GPU, a 24GB desktop GPU). The measured `ms` is logged so the
// thresholds can be tuned to fit; until then these are educated guesses.
function classify(ms: number): PerfClass {
    if (ms < 8) return "fast";
    if (ms < 30) return "mid";
    return "slow";
}

interface CachedShape { v: number; fp: string; ms: number; }

function fingerprint(info?: { vendor?: string; architecture?: string; description?: string }): string {
    return [info?.vendor, info?.architecture, info?.description].filter(Boolean).join("|") || "unknown";
}

function readCache(fp: string): number | null {
    try {
        const raw = localStorage.getItem(CACHE_KEY);
        if (!raw) return null;
        const c = JSON.parse(raw) as CachedShape;
        if (c.v === BENCH_VERSION && c.fp === fp && Number.isFinite(c.ms)) return c.ms;
    } catch { /* unparseable / unavailable */ }
    return null;
}

function writeCache(fp: string, ms: number): void {
    try {
        localStorage.setItem(CACHE_KEY, JSON.stringify({ v: BENCH_VERSION, fp, ms } satisfies CachedShape));
    } catch { /* private mode / quota */ }
}

// GPUBufferUsage flag bits. Hardcoded because the WebGPU *value* globals
// aren't in scope for tsc (no other TS file creates GPU buffers — the engine
// does that in Rust/wasm). These are the spec-fixed constants.
const BUF_STORAGE  = 0x0080;
const BUF_COPY_DST = 0x0008;
const BUF_UNIFORM  = 0x0040;

const MATMUL_WGSL = /* wgsl */`
struct Dims { n: u32 };
@group(0) @binding(0) var<storage, read>        A : array<f32>;
@group(0) @binding(1) var<storage, read>        B : array<f32>;
@group(0) @binding(2) var<storage, read_write>  C : array<f32>;
@group(0) @binding(3) var<uniform>              d : Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let n = d.n;
    let row = gid.y;
    let col = gid.x;
    if (row >= n || col >= n) { return; }
    var acc = 0.0;
    for (var k = 0u; k < n; k = k + 1u) {
        acc = acc + A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = acc;
}
`;

/**
 * Run the GPU burst once on a throwaway device and return the wall-clock ms
 * for REPS matmuls. Never throws — returns null on any failure (no WebGPU,
 * device-creation refusal, etc.); callers treat null as "no perf signal".
 */
async function runBurst(): Promise<{ ms: number; fp: string } | null> {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    if (!gpu) return null;
    let device: GPUDevice | null = null;
    try {
        const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
        if (!adapter) return null;
        const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
        const fp = fingerprint(info);
        device = await adapter.requestDevice();

        const n = DIM;
        const bytes = n * n * 4;
        const mk = (usage: number) => device!.createBuffer({ size: bytes, usage });
        const A = mk(BUF_STORAGE | BUF_COPY_DST);
        const B = mk(BUF_STORAGE | BUF_COPY_DST);
        const C = mk(BUF_STORAGE);
        // Seed A/B with 1.0 so the result is finite (values don't matter).
        const seed = new Float32Array(n * n).fill(1);
        device.queue.writeBuffer(A, 0, seed);
        device.queue.writeBuffer(B, 0, seed);
        const dims = device.createBuffer({ size: 16, usage: BUF_UNIFORM | BUF_COPY_DST });
        device.queue.writeBuffer(dims, 0, new Uint32Array([n, 0, 0, 0]));

        const module = device.createShaderModule({ code: MATMUL_WGSL });
        const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
        const bind = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: A } },
                { binding: 1, resource: { buffer: B } },
                { binding: 2, resource: { buffer: C } },
                { binding: 3, resource: { buffer: dims } },
            ],
        });
        const groups = Math.ceil(n / 16);
        const dispatch = (reps: number) => {
            const enc = device!.createCommandEncoder();
            for (let r = 0; r < reps; r++) {
                const pass = enc.beginComputePass();
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, bind);
                pass.dispatchWorkgroups(groups, groups);
                pass.end();
            }
            device!.queue.submit([enc.finish()]);
        };

        // Warmup (untimed) — eats pipeline/shader compilation + first-submit cost.
        dispatch(2);
        await device.queue.onSubmittedWorkDone();

        const t0 = performance.now();
        dispatch(REPS);
        await device.queue.onSubmittedWorkDone();
        const ms = performance.now() - t0;

        return { ms, fp };
    } catch {
        return null;
    } finally {
        try { device?.destroy(); } catch { /* */ }
    }
}

/**
 * Get the GPU performance class, running the burst at most ONCE per device
 * and caching the result. Returns null when no perf signal is available
 * (no WebGPU / burst failed) — callers fall back to the static tier alone.
 */
export async function getGpuBench(): Promise<BenchResult | null> {
    // Fast path: a cached number for this adapter fingerprint. We still need
    // the fingerprint, but resolving it is cheap (adapter request, no device).
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    let fp = "unknown";
    if (gpu) {
        try {
            const adapter = await gpu.requestAdapter();
            const info = adapter && (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
            fp = fingerprint(info ?? undefined);
        } catch { /* keep "unknown" */ }
    }
    const cached = readCache(fp);
    if (cached != null) return { ms: cached, perfClass: classify(cached), fresh: false };

    const res = await runBurst();
    if (!res) return null;
    writeCache(res.fp, res.ms);
    return { ms: res.ms, perfClass: classify(res.ms), fresh: true };
}
