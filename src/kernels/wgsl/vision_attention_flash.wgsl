// Flash-attention-style bidirectional self-attention for the Gemma 4
// vision tower. Same I/O contract as vision_attention.wgsl:
//
//   q, k, v: f32 [n_patches, n_heads, head_dim]
//   out:     f32 [n_patches, n_heads, head_dim]
//
// Dispatch:  (n_patches, n_heads, 1)  — one workgroup per (query, head).
// Workgroup: 64 threads, head_dim = 64 → 1 thread per output dim.
//
// Replaces vision_attention.wgsl whose Phase E had each thread doing a
// 2304-iter sequential inner loop reading V from global memory (5 s per
// layer at 768×768 input on Radeon Pro 555). This kernel:
//
//   1. Loads q into workgroup-shared cache (64 elements, 4 bytes each).
//   2. Loops over patches in tiles of TILE_T = 32.
//   3. For each tile:
//      a. Cooperatively load K_tile (32 × 64 = 8 KB) into a shared buffer
//         (`kv_tile`).
//      b. Each thread computes one score = q · K_tile[t_local, *].
//      c. Tile-wide max-reduce → online softmax merge with the running
//         max `m` and normalizer `l`. Each thread rescales its output
//         accumulator `o` by `exp(m_old - m_new)`.
//      d. Reuse `kv_tile` for V — cooperative load of V_tile (32 × 64).
//      e. Each thread accumulates `o += Σ_t scores[t] · V_tile[t, tid]`.
//   4. After the last tile, normalize: `out[tid] = o / l`.
//
// Workgroup storage:
//   q_shared    (64 f32)      256 B
//   kv_tile     (32×64 f32)  8192 B  (re-used for K then V per tile)
//   tile_scores (64 f32)      256 B
//   rbuf        (64 f32)      256 B
//   --------------------------------
//   total                    8960 B  (fits in WebGPU's 16 KB minimum)
//
// Assumes head_dim ≤ 64. TILE_T = 32 chosen for high CU occupancy on
// AMD Radeon Pro 555 (64 KB LDS / CU). Bumping to TILE_T=64 cuts barrier
// count in half but doubles workgroup-shared size, dropping occupancy
// from ~7 to ~3 workgroups per CU — measured 2× regression on the Pro 555.

struct Params {
    head_dim:  u32,
    n_heads:   u32,
    n_patches: u32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       q:      array<f32>;
@group(0) @binding(2) var<storage, read>       k:      array<f32>;
@group(0) @binding(3) var<storage, read>       v:      array<f32>;
@group(0) @binding(4) var<storage, read_write> out:    array<f32>;

const WG: u32 = 64u;
const HEAD_DIM_MAX: u32 = 64u;
const TILE_T: u32 = 32u;

var<workgroup> q_shared:    array<f32, HEAD_DIM_MAX>;
var<workgroup> kv_tile:     array<f32, 2048>;          // TILE_T × HEAD_DIM_MAX, reused for K then V
// Sized WG (not TILE_T) so threads with tid ≥ TILE_T can safely write -inf
// into their slot without OOB. Only the first TILE_T slots ever feed into
// the weighted sum.
var<workgroup> tile_scores: array<f32, WG>;
var<workgroup> rbuf:        array<f32, WG>;

fn block_max_reduce(tid: u32) -> f32 {
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            rbuf[tid] = max(rbuf[tid], rbuf[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return rbuf[0];
}

fn block_sum_reduce(tid: u32) -> f32 {
    var stride: u32 = WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            rbuf[tid] = rbuf[tid] + rbuf[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    return rbuf[0];
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id)         wid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let bq: u32 = wid.x;
    let qh: u32 = wid.y;
    if (bq >= params.n_patches || qh >= params.n_heads) { return; }

    let head_dim:  u32 = params.head_dim;
    let n_patches: u32 = params.n_patches;
    let n_heads:   u32 = params.n_heads;

    // --- Load q into workgroup shared memory (one element per thread). ---
    let q_off: u32 = (bq * n_heads + qh) * head_dim;
    if (tid < head_dim) {
        q_shared[tid] = q[q_off + tid];
    }
    workgroupBarrier();

    // Online softmax state. `o` is one element of the output vector per thread.
    var m: f32 = -1.0e30;
    var l: f32 = 0.0;
    var o: f32 = 0.0;

    let n_tiles = (n_patches + TILE_T - 1u) / TILE_T;
    for (var tile: u32 = 0u; tile < n_tiles; tile = tile + 1u) {
        let t0 = tile * TILE_T;
        let tile_size = min(TILE_T, n_patches - t0);

        // --- Load K_tile into kv_tile (tile_size × head_dim). ---
        // Each thread loads (tile_size * head_dim + WG - 1) / WG slots.
        let total_k = tile_size * head_dim;
        var lk = tid;
        loop {
            if (lk >= total_k) { break; }
            let t_local = lk / head_dim;
            let d_local = lk % head_dim;
            let g_off = ((t0 + t_local) * n_heads + qh) * head_dim + d_local;
            kv_tile[lk] = k[g_off];
            lk = lk + WG;
        }
        workgroupBarrier();

        // --- Score per t_local. Thread tid handles t_local = tid (or none if past). ---
        var s_t: f32 = -1.0e30;
        if (tid < tile_size) {
            var sum: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                sum = sum + q_shared[d] * kv_tile[tid * head_dim + d];
            }
            s_t = sum;
        }
        tile_scores[tid] = s_t;
        workgroupBarrier();

        // --- Tile-wide max. ---
        rbuf[tid] = s_t;
        workgroupBarrier();
        let tile_m = block_max_reduce(tid);

        // --- Online softmax merge. ---
        let m_new = max(m, tile_m);
        let alpha = exp(m - m_new);

        var p_t: f32 = 0.0;
        if (tid < tile_size) {
            p_t = exp(s_t - m_new);
        }
        tile_scores[tid] = p_t;

        rbuf[tid] = p_t;
        workgroupBarrier();
        let tile_sum = block_sum_reduce(tid);

        l = l * alpha + tile_sum;
        m = m_new;

        // Rescale the running output accumulator BEFORE summing the new tile's
        // V contribution (so partial sums all share the new max).
        o = o * alpha;

        // --- Reuse kv_tile for V. Load V_tile. ---
        workgroupBarrier();   // ensure all threads finished reading kv_tile (K)
        var lv = tid;
        loop {
            if (lv >= total_k) { break; }
            let t_local = lv / head_dim;
            let d_local = lv % head_dim;
            let g_off = ((t0 + t_local) * n_heads + qh) * head_dim + d_local;
            kv_tile[lv] = v[g_off];
            lv = lv + WG;
        }
        workgroupBarrier();

        // --- Weighted sum: each thread adds its column of V scaled by scores. ---
        if (tid < head_dim) {
            var contrib: f32 = 0.0;
            for (var t_local: u32 = 0u; t_local < tile_size; t_local = t_local + 1u) {
                contrib = contrib + tile_scores[t_local] * kv_tile[t_local * head_dim + tid];
            }
            o = o + contrib;
        }
        workgroupBarrier();
    }

    // --- Normalize and write out. ---
    if (tid < head_dim) {
        let out_off = (bq * n_heads + qh) * head_dim + tid;
        out[out_off] = o / l;
    }
}
