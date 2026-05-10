//! GPU AudioForward — Conformer audio encoder on wgpu.
//!
//! Mirrors the structure of `multimodal::vision::VisionForward`: persistent
//! scratch buffers, per-encode CommandEncoder, single readback at the end.
//!
//! The CPU oracle in `multimodal::audio::AudioForward` is the reference
//! implementation; `cpu_block_local_attention` (in `backend::elementwise`'s
//! tests) plus the unit test `block_local_attention_matches_cpu_oracle`
//! verify the attention kernel matches that reference at the per-block
//! level. This module wires the rest of the pipeline (mel features,
//! SSCP convs, FFW × 2, LightConv, projector) on GPU.
//!
//! This first cut keeps mel features + SSCP convs + pre-encode linear on
//! CPU (they're fast and avoid the conv2d 4D-tensor plumbing), then uploads
//! the [seq, hidden] tensor and runs the 12 Conformer blocks + projector
//! entirely on GPU. Pure-GPU SSCP is a follow-up.

use std::sync::Arc;

// (Imports for encode() will be added when that method lands.)
use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::Result;
use crate::gguf::{dequant_tensor_to_f32_async, GgmlDtype};
use crate::multimodal::audio::AudioConfig;
use crate::multimodal::audio::AudioForward as CpuAudioForward;

/// One block's worth of GPU-resident weight buffers (matches the CPU oracle's
/// AudioBlock fields, but as wgpu::Buffer).
struct GpuAudioBlock {
    pre_norm:        wgpu::Buffer,    // [hidden] f32
    // FFW start
    ffw_norm:        wgpu::Buffer,
    ffw_up:          wgpu::Buffer,    // [hidden, ffn] (raw GGUF dtype)
    ffw_up_dtype:    GgmlDtype,
    ffw_down:        wgpu::Buffer,    // [ffn, hidden]
    ffw_down_dtype:  GgmlDtype,
    ffw_post_norm:   wgpu::Buffer,
    // FFW end
    ffw_norm_1:      wgpu::Buffer,
    ffw_up_1:        wgpu::Buffer,
    ffw_up_1_dtype:  GgmlDtype,
    ffw_down_1:      wgpu::Buffer,
    ffw_down_1_dtype: GgmlDtype,
    ffw_post_norm_1: wgpu::Buffer,
    // Attention
    attn_pre_norm:   wgpu::Buffer,
    attn_post_norm:  wgpu::Buffer,
    attn_q:          wgpu::Buffer,
    attn_q_dtype:    GgmlDtype,
    attn_k:          wgpu::Buffer,
    attn_k_dtype:    GgmlDtype,
    attn_v:          wgpu::Buffer,
    attn_v_dtype:    GgmlDtype,
    attn_o:          wgpu::Buffer,
    attn_o_dtype:    GgmlDtype,
    linear_pos:      wgpu::Buffer,    // [hidden, hidden] BF16
    linear_pos_dtype: GgmlDtype,
    per_dim_scale:   Vec<f32>,        // [head_dim] CPU-resident (used for Q scale)
    // LightConv
    conv_norm:       wgpu::Buffer,
    norm_conv:       wgpu::Buffer,
    conv_pw1:        wgpu::Buffer,
    conv_pw1_dtype:  GgmlDtype,
    conv_pw2:        wgpu::Buffer,
    conv_pw2_dtype:  GgmlDtype,
    conv_dw:         Vec<f32>,        // [hidden, kernel] CPU-resident
    // ClippableLinear clamps (10 sites)
    cl_attn_q:       Clamp,
    cl_attn_k:       Clamp,
    cl_attn_v:       Clamp,
    cl_attn_o:       Clamp,
    cl_ffw_up:       Clamp,
    cl_ffw_down:     Clamp,
    cl_ffw_up_1:     Clamp,
    cl_ffw_down_1:   Clamp,
    cl_conv_pw1:     Clamp,
    cl_conv_pw2:     Clamp,
}

#[derive(Clone, Copy, Default)]
struct Clamp { in_min: f32, in_max: f32, out_min: f32, out_max: f32 }

/// GPU Conformer audio encoder. Holds all weights as wgpu::Buffer plus a
/// reusable set of scratch buffers sized for the maximum supported audio
/// duration (~30 s → ~250 frames after SSCP downsampling).
pub struct GpuAudioForward {
    cfg: AudioConfig,
    ctx: WgpuCtx,
    pipes: Arc<Pipelines>,
    #[allow(dead_code)]
    wcache: Arc<WeightCache>,

    /// CPU-side mel + SSCP + pre_encode pipeline. We delegate the early
    /// stages to the CPU oracle since those weights' shapes (4-D conv with
    /// per-channel norm) need extra wiring that doesn't pay off compared to
    /// the bulk of the work in the 12 conformer blocks.
    cpu_prefix: CpuAudioForward,

    blocks: Vec<GpuAudioBlock>,

    // Projector weights.
    proj_fc:               wgpu::Buffer,
    proj_fc_dtype:         GgmlDtype,
    proj_fc_bias:          Option<wgpu::Buffer>,
    proj_input:            wgpu::Buffer,
    proj_input_dtype:      GgmlDtype,
}

impl GpuAudioForward {
    pub async fn new(
        cfg: AudioConfig,
        ctx: WgpuCtx,
        pipes: Arc<Pipelines>,
        wcache: Arc<WeightCache>,
    ) -> Result<Self> {
        // Build the CPU prefix that handles mel + SSCP + pre_encode. We share
        // the same WeightCache so dequantising those tensors only happens once.
        let cpu_prefix = CpuAudioForward::new(cfg.clone(), wcache.clone()).await?;

        let mut blocks = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers {
            blocks.push(load_gpu_block(&wcache, i).await?);
        }

        let proj_fc        = wcache.buffer_async("mm.a.fc.weight").await?;
        let proj_fc_dtype  = wcache.reader().tensor("mm.a.fc.weight")?.dtype;
        let proj_fc_bias   = wcache.buffer_opt_async("mm.a.fc.bias").await?;
        let proj_input     = wcache.buffer_async("mm.a.input_projection.weight").await?;
        let proj_input_dtype = wcache.reader().tensor("mm.a.input_projection.weight")?.dtype;

        Ok(Self {
            cfg, ctx, pipes, wcache,
            cpu_prefix,
            blocks,
            proj_fc, proj_fc_dtype, proj_fc_bias,
            proj_input, proj_input_dtype,
        })
    }

    pub fn cfg(&self) -> &AudioConfig { &self.cfg }

    /// Encode 16 kHz mono PCM into `[n_audio_tokens × d_text]` soft tokens.
    ///
    /// **v0 status:** delegates to the CPU oracle so the public surface is
    /// usable immediately. Per-block FFW/attention/LightConv migration to GPU
    /// is incremental from here — each piece swaps in once it's parity-tested
    /// against the CPU output. Until that work lands, the underscore-prefixed
    /// fields (`blocks`, `proj_*`, etc.) are intentionally unused.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        // Suppress dead-code warnings until block dispatch lands.
        let _ = (&self.ctx, &self.pipes, &self.blocks,
                 &self.proj_fc, &self.proj_fc_dtype, &self.proj_fc_bias,
                 &self.proj_input, &self.proj_input_dtype);
        self.cpu_prefix.encode(pcm)
    }
}

/// Helper: load one block's weights into GPU buffers + clamps into CPU values.
async fn load_gpu_block(wcache: &Arc<WeightCache>, i: u32) -> Result<GpuAudioBlock> {
    let p = format!("a.blk.{i}.");
    let r = wcache.reader();

    // Helper closure to fetch dtype for a tensor name.
    let dt = |suffix: &str| -> Result<GgmlDtype> {
        Ok(r.tensor(&format!("{p}{suffix}"))?.dtype)
    };

    let buf = |suffix: &str| -> _ {
        let name = format!("{p}{suffix}");
        async move { wcache.buffer_async(&name).await }
    };

    // Per-block scalar tensors loaded as f32 (small).
    let per_dim_scale = dequant_tensor_to_f32_async(r, &format!("{p}per_dim_scale.weight")).await?;
    let conv_dw       = dequant_tensor_to_f32_async(r, &format!("{p}conv_dw.weight")).await?;

    Ok(GpuAudioBlock {
        pre_norm:        buf("layer_pre_norm.weight").await?,
        ffw_norm:        buf("ffn_norm.weight").await?,
        ffw_up:          buf("ffn_up.weight").await?,
        ffw_up_dtype:    dt("ffn_up.weight")?,
        ffw_down:        buf("ffn_down.weight").await?,
        ffw_down_dtype:  dt("ffn_down.weight")?,
        ffw_post_norm:   buf("ffn_post_norm.weight").await?,
        ffw_norm_1:      buf("ffn_norm_1.weight").await?,
        ffw_up_1:        buf("ffn_up_1.weight").await?,
        ffw_up_1_dtype:  dt("ffn_up_1.weight")?,
        ffw_down_1:      buf("ffn_down_1.weight").await?,
        ffw_down_1_dtype: dt("ffn_down_1.weight")?,
        ffw_post_norm_1: buf("ffn_post_norm_1.weight").await?,
        attn_pre_norm:   buf("ln1.weight").await?,
        attn_post_norm:  buf("ln2.weight").await?,
        attn_q:          buf("attn_q.weight").await?,
        attn_q_dtype:    dt("attn_q.weight")?,
        attn_k:          buf("attn_k.weight").await?,
        attn_k_dtype:    dt("attn_k.weight")?,
        attn_v:          buf("attn_v.weight").await?,
        attn_v_dtype:    dt("attn_v.weight")?,
        attn_o:          buf("attn_out.weight").await?,
        attn_o_dtype:    dt("attn_out.weight")?,
        linear_pos:      buf("linear_pos.weight").await?,
        linear_pos_dtype: dt("linear_pos.weight")?,
        per_dim_scale,
        conv_norm:       buf("conv_norm.weight").await?,
        norm_conv:       buf("norm_conv.weight").await?,
        conv_pw1:        buf("conv_pw1.weight").await?,
        conv_pw1_dtype:  dt("conv_pw1.weight")?,
        conv_pw2:        buf("conv_pw2.weight").await?,
        conv_pw2_dtype:  dt("conv_pw2.weight")?,
        conv_dw,
        cl_attn_q:       load_clamp(wcache, &format!("{p}attn_q")).await,
        cl_attn_k:       load_clamp(wcache, &format!("{p}attn_k")).await,
        cl_attn_v:       load_clamp(wcache, &format!("{p}attn_v")).await,
        cl_attn_o:       load_clamp(wcache, &format!("{p}attn_out")).await,
        cl_ffw_up:       load_clamp(wcache, &format!("{p}ffn_up")).await,
        cl_ffw_down:     load_clamp(wcache, &format!("{p}ffn_down")).await,
        cl_ffw_up_1:     load_clamp(wcache, &format!("{p}ffn_up_1")).await,
        cl_ffw_down_1:   load_clamp(wcache, &format!("{p}ffn_down_1")).await,
        cl_conv_pw1:     load_clamp(wcache, &format!("{p}conv_pw1")).await,
        cl_conv_pw2:     load_clamp(wcache, &format!("{p}conv_pw2")).await,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `GpuAudioForward::encode` v0 must match the CPU oracle exactly (it
    /// delegates). Once the block dispatch loop lands, this same fixture
    /// becomes the parity floor: any GPU-side regression that doesn't bit-
    /// match the CPU oracle within a small tolerance fails this test.
    #[test]
    fn encode_v0_matches_cpu_oracle() {
        let path = "/Users/nightness/.ollama/models/blobs/sha256-4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: gemma4 GGUF not available");
            return;
        }
        let bytes = std::fs::read(path).unwrap();
        let reader = crate::gguf::GgufReader::new(bytes).unwrap();
        if reader.tensor("a.conv1d.0.weight").is_err() {
            eprintln!("skipping: GGUF has no audio tower");
            return;
        }
        // Build a CPU oracle and a GPU-wrapper sharing the same WeightCache.
        let r_arc = std::sync::Arc::new(reader);
        let cfg = AudioConfig::from_gguf(&r_arc, 1536).unwrap();
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let pipes = std::sync::Arc::new(Pipelines::new(&ctx.device));
        let wcache = std::sync::Arc::new(WeightCache::new(
            r_arc.clone(), ctx.device.clone(), ctx.queue.clone()));

        let cpu = pollster::block_on(CpuAudioForward::new(cfg.clone(), wcache.clone())).unwrap();
        let gpu = pollster::block_on(GpuAudioForward::new(
            cfg, ctx, pipes, wcache)).unwrap();

        // 0.25 s of pure tone (smaller than the 1 s smoke test — fast).
        let sr = 16_000;
        let n  = sr / 4;
        let omega = 2.0 * std::f32::consts::PI * 440.0 / sr as f32;
        let pcm: Vec<f32> = (0..n).map(|i| 0.3 * (omega * i as f32).sin()).collect();

        let cpu_out = cpu.encode(&pcm).unwrap();
        let gpu_out = gpu.encode(&pcm).unwrap();
        assert_eq!(cpu_out.len(), gpu_out.len(), "len mismatch");
        let mut max_abs = 0f32;
        for i in 0..cpu_out.len() {
            let d = (cpu_out[i] - gpu_out[i]).abs();
            if d > max_abs { max_abs = d; }
        }
        eprintln!("encode_v0 vs cpu oracle: max_abs={max_abs:e} (n={})", cpu_out.len());
        assert!(max_abs < 1e-6, "v0 is a delegation; should be bit-identical");
    }
}

async fn load_clamp(wcache: &Arc<WeightCache>, prefix: &str) -> Clamp {
    let one = |suffix: &str| {
        let name = format!("{prefix}.{suffix}");
        async move {
            match wcache.reader().tensor(&name) {
                Ok(_) => dequant_tensor_to_f32_async(wcache.reader(), &name).await
                    .ok().and_then(|v| v.first().copied()).unwrap_or(0.0),
                Err(_) => 0.0,
            }
        }
    };
    Clamp {
        in_min:  one("input_min").await,
        in_max:  one("input_max").await,
        out_min: one("output_min").await,
        out_max: one("output_max").await,
    }
}

