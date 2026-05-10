//! Cached compute pipelines for the full forward pass.
//!
//! Built once per [`Backend`] (i.e., once per model load). Pipeline / shader-module
//! creation is expensive (tens to hundreds of milliseconds in the browser); a 35-layer
//! Gemma 4 forward dispatches dozens of compute calls per layer × hundreds of tokens,
//! so amortizing this cost is the difference between "one-shot demo" and "interactive".

use std::borrow::Cow;

use crate::kernels;

pub struct Pipelines {
    pub f16_matmul:    wgpu::ComputePipeline,
    pub q4_k_matmul:   wgpu::ComputePipeline,
    pub q6_k_matmul:   wgpu::ComputePipeline,
    pub rmsnorm:       wgpu::ComputePipeline,
    pub softcap:       wgpu::ComputePipeline,
    pub geglu:         wgpu::ComputePipeline,
    pub rope_neox:     wgpu::ComputePipeline,
    pub attention:     wgpu::ComputePipeline,
    pub residual_add:      wgpu::ComputePipeline,
    pub scale:             wgpu::ComputePipeline,
    pub rmsnorm_per_row:   wgpu::ComputePipeline,
    pub q4_k_matmul_tiled: wgpu::ComputePipeline,
    pub q6_k_matmul_tiled: wgpu::ComputePipeline,
    pub conv2d:            wgpu::ComputePipeline,
    pub avg_pool2d:        wgpu::ComputePipeline,
    pub clamp:             wgpu::ComputePipeline,
    pub quick_geglu:       wgpu::ComputePipeline,
    pub rope_2d:           wgpu::ComputePipeline,
    pub f16_matmul_batched: wgpu::ComputePipeline,
    pub f16_matmul_batched_tiled: wgpu::ComputePipeline,
    pub pos_embed_add:     wgpu::ComputePipeline,
    pub vision_attention:  wgpu::ComputePipeline,
    pub half_residual_add: wgpu::ComputePipeline,
    pub silu:              wgpu::ComputePipeline,
    pub glu_split:         wgpu::ComputePipeline,
    pub depthwise_conv1d:  wgpu::ComputePipeline,
    pub block_local_attention: wgpu::ComputePipeline,
    pub bf16_matmul:       wgpu::ComputePipeline,
    pub bf16_matmul_batched: wgpu::ComputePipeline,
    pub bf16_matmul_batched_tiled: wgpu::ComputePipeline,
    pub scale_per_inner_dim: wgpu::ComputePipeline,
    pub add_bias_batched: wgpu::ComputePipeline,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            f16_matmul:        build(device, "f16_matmul",        kernels::F16_MATMUL),
            q4_k_matmul:       build(device, "q4_k_matmul",       kernels::Q4_K_DEQUANT_MATMUL),
            q6_k_matmul:       build(device, "q6_k_matmul",       kernels::Q6_K_DEQUANT_MATMUL),
            rmsnorm:           build(device, "rmsnorm",           kernels::RMSNORM),
            softcap:           build(device, "softcap",           kernels::SOFTCAP),
            geglu:             build(device, "geglu",             kernels::GEGLU),
            rope_neox:         build(device, "rope_neox",         kernels::ROPE_NEOX),
            attention:         build(device, "attention",         kernels::ATTENTION),
            residual_add:      build(device, "residual_add",      kernels::RESIDUAL_ADD),
            scale:             build(device, "scale",             kernels::SCALE),
            rmsnorm_per_row:   build(device, "rmsnorm_per_row",   kernels::RMSNORM_PER_ROW),
            q4_k_matmul_tiled: build(device, "q4_k_matmul_tiled", kernels::Q4_K_DEQUANT_MATMUL_TILED),
            q6_k_matmul_tiled: build(device, "q6_k_matmul_tiled", kernels::Q6_K_DEQUANT_MATMUL_TILED),
            conv2d:            build(device, "conv2d",            kernels::CONV2D),
            avg_pool2d:        build(device, "avg_pool2d",        kernels::AVG_POOL2D),
            clamp:             build(device, "clamp",             kernels::CLAMP),
            quick_geglu:       build(device, "quick_geglu",       kernels::QUICK_GEGLU),
            rope_2d:           build(device, "rope_2d",           kernels::ROPE_2D),
            f16_matmul_batched: build(device, "f16_matmul_batched", kernels::F16_MATMUL_BATCHED),
            f16_matmul_batched_tiled: build(device, "f16_matmul_batched_tiled", kernels::F16_MATMUL_BATCHED_TILED),
            pos_embed_add:     build(device, "pos_embed_add",     kernels::POS_EMBED_ADD),
            vision_attention:  build(device, "vision_attention",  kernels::VISION_ATTENTION),
            half_residual_add: build(device, "half_residual_add", kernels::HALF_RESIDUAL_ADD),
            silu:              build(device, "silu",              kernels::SILU),
            glu_split:         build(device, "glu_split",         kernels::GLU_SPLIT),
            depthwise_conv1d:  build(device, "depthwise_conv1d",  kernels::DEPTHWISE_CONV1D),
            block_local_attention: build(device, "block_local_attention", kernels::BLOCK_LOCAL_ATTENTION),
            bf16_matmul:       build(device, "bf16_matmul",       kernels::BF16_MATMUL),
            bf16_matmul_batched: build(device, "bf16_matmul_batched", kernels::BF16_MATMUL_BATCHED),
            bf16_matmul_batched_tiled: build(device, "bf16_matmul_batched_tiled", kernels::BF16_MATMUL_BATCHED_TILED),
            scale_per_inner_dim: build(device, "scale_per_inner_dim", kernels::SCALE_PER_INNER_DIM),
            add_bias_batched: build(device, "add_bias_batched", kernels::ADD_BIAS_BATCHED),
        }
    }
}

fn build(device: &wgpu::Device, label: &str, wgsl: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label}.module")),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}.pipeline")),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
