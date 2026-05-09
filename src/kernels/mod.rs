//! WGSL kernels — included as static strings via `include_str!`.
//!
//! Each kernel is paired with a typed Rust dispatcher in [`crate::backend::matmul`]
//! that creates the pipeline, uploads inputs, dispatches, and reads back results.

pub const ATTENTION: &str = include_str!("wgsl/attention.wgsl");
pub const AVG_POOL2D: &str = include_str!("wgsl/avg_pool2d.wgsl");
pub const CLAMP: &str = include_str!("wgsl/clamp.wgsl");
pub const CONV2D: &str = include_str!("wgsl/conv2d.wgsl");
pub const F16_MATMUL: &str = include_str!("wgsl/f16_matmul.wgsl");
pub const F16_MATMUL_BATCHED: &str = include_str!("wgsl/f16_matmul_batched.wgsl");
pub const QUICK_GEGLU: &str = include_str!("wgsl/quick_geglu.wgsl");
pub const POS_EMBED_ADD: &str = include_str!("wgsl/pos_embed_add.wgsl");
pub const ROPE_2D: &str = include_str!("wgsl/rope_2d.wgsl");
pub const VISION_ATTENTION: &str = include_str!("wgsl/vision_attention.wgsl");
pub const GEGLU: &str = include_str!("wgsl/geglu.wgsl");
pub const Q4_K_DEQUANT_MATMUL: &str = include_str!("wgsl/q4_k_dequant_matmul.wgsl");
pub const Q4_K_DEQUANT_MATMUL_TILED: &str = include_str!("wgsl/q4_k_dequant_matmul_tiled.wgsl");
pub const Q6_K_DEQUANT_MATMUL: &str = include_str!("wgsl/q6_k_dequant_matmul.wgsl");
pub const Q6_K_DEQUANT_MATMUL_TILED: &str = include_str!("wgsl/q6_k_dequant_matmul_tiled.wgsl");
pub const RESIDUAL_ADD: &str = include_str!("wgsl/residual_add.wgsl");
pub const RMSNORM: &str = include_str!("wgsl/rmsnorm.wgsl");
pub const RMSNORM_PER_ROW: &str = include_str!("wgsl/rmsnorm_per_row.wgsl");
pub const ROPE_NEOX: &str = include_str!("wgsl/rope_neox.wgsl");
pub const SCALE: &str = include_str!("wgsl/scale.wgsl");
pub const SOFTCAP: &str = include_str!("wgsl/softcap.wgsl");
