// Batched BF16 weight matmul: y[b, j] = Σ_i x[b, i] * W[j, i].
//
// Mirrors f16_matmul_batched.wgsl with BF16 weights. BF16 is just the upper
// 16 bits of an F32, so each value reconstructs as
// `bitcast<f32>(u32(bf16) << 16)`.
//
// Used by the audio Conformer tower so each block linear processes all
// `seq` frames in a single dispatch instead of `seq` separate ones.

struct Params {
    k:     u32,
    n:     u32,
    batch: u32,
    _pad:  u32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<storage, read>       weight: array<u32>;   // bf16 pairs per u32
@group(0) @binding(2) var<storage, read>       x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> y:      array<f32>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

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
        let lo: f32 = bf16_to_f32(packed & 0x0000FFFFu);
        let hi: f32 = bf16_to_f32(packed >> 16u);
        acc = acc + x[x_off + p * 2u] * lo + x[x_off + p * 2u + 1u] * hi;
    }
    y[idx] = acc;
}
