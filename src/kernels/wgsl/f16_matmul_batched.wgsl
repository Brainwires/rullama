// Batched f16-weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i].
//
// Same W layout as f16_matmul.wgsl (n rows of length k, f16 packed two per u32),
// but x is now [batch, k] and y is [batch, n]. Used by the vision tower where
// each call processes all `num_patches` rows simultaneously — a single dispatch
// per linear instead of `num_patches` separate dispatches.
//
// One thread per output element, indexed by gid.x = batch * n + j.

struct Params {
    k:     u32,
    n:     u32,
    batch: u32,
    _pad:  u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;   // f16 pairs per u32
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total: u32 = params.batch * params.n;
    let idx: u32 = gid.x;
    if (idx >= total) { return; }

    let b: u32 = idx / params.n;
    let j: u32 = idx - b * params.n;

    let half_k: u32 = params.k / 2u;
    let row_start: u32 = j * half_k;
    let x_off: u32 = b * params.k;

    var acc: f32 = 0.0;
    for (var p: u32 = 0u; p < half_k; p = p + 1u) {
        let packed: u32 = weight[row_start + p];
        let pair: vec2<f32> = unpack2x16float(packed);
        acc = acc + x[x_off + p * 2u] * pair.x + x[x_off + p * 2u + 1u] * pair.y;
    }
    y[idx] = acc;
}
