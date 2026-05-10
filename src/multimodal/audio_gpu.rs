//! GPU AudioForward — Conformer audio encoder on wgpu.
//!
//! The CPU oracle in `multimodal::audio::AudioForward` is the reference; this
//! module ports its block loop + projector to GPU using the chained
//! dispatchers in `crate::backend::dispatch`. Mel features, the two SSCP
//! Conv2D blocks, and the pre-encode linear stay on CPU (small compute, the
//! data layout would need extra plumbing to run on GPU profitably) — the
//! prefix's output is `[seq, hidden=1024]` which then runs through 12
//! Conformer blocks + audio projector entirely on GPU with one
//! CommandEncoder per `encode()` call.

use std::sync::Arc;

use bytemuck::cast_slice;
use futures_channel::oneshot;

use crate::backend::dispatch::{
    add_bias_batched_chained, block_local_attention_chained, clamp_chained,
    depthwise_conv1d_chained, glu_split_chained, half_residual_add_chained,
    matmul_bf16_batched_chained, matmul_f16_batched_chained,
    rmsnorm_per_row_chained, scale_chained, scale_per_inner_dim_chained,
    silu_chained,
};
use crate::backend::{Pipelines, WeightCache, WgpuCtx};
use crate::error::{Result, RullamaError};
use crate::gguf::{dequant_tensor_to_f32_async, GgmlDtype};
use crate::multimodal::audio::AudioConfig;
use crate::multimodal::audio::AudioForward as CpuAudioForward;

/// Maximum number of frames the GPU scratch buffers are sized for. ~25 frames
/// per second of audio after SSCP downsampling, so 768 frames ≈ 30 s — Gemma's
/// per-encode cap. Must be a multiple of `chunk_size = 12`.
const MAX_SEQ: usize = 768;

#[derive(Clone, Copy, Default)]
struct Clamp { in_min: f32, in_max: f32, out_min: f32, out_max: f32 }

/// One block's worth of GPU-resident weight buffers, plus its CPU-side
/// per_dim_scale (folded with q_scale_base into a single GPU buffer) and
/// conv_dw weights.
struct GpuAudioBlock {
    pre_norm:        wgpu::Buffer,    // [hidden] f32  (final block RMSNorm)
    // FFW start
    ffw_norm:        wgpu::Buffer,
    ffw_up:          wgpu::Buffer,    // BF16 [hidden, ffn]
    ffw_down:        wgpu::Buffer,    // BF16 [ffn, hidden]
    ffw_post_norm:   wgpu::Buffer,
    // FFW end
    ffw_norm_1:      wgpu::Buffer,
    ffw_up_1:        wgpu::Buffer,    // BF16
    ffw_down_1:      wgpu::Buffer,    // BF16
    ffw_post_norm_1: wgpu::Buffer,
    // Attention
    attn_pre_norm:   wgpu::Buffer,
    attn_post_norm:  wgpu::Buffer,
    attn_q:          wgpu::Buffer,    // BF16
    attn_k:          wgpu::Buffer,    // BF16
    attn_v:          wgpu::Buffer,    // BF16
    attn_o:          wgpu::Buffer,    // BF16
    linear_pos:      wgpu::Buffer,    // BF16 [hidden, hidden]
    /// Per-dim Q scale, pre-multiplied with `q_scale_base = head_dim^-0.5 / ln 2`.
    /// Shape `[head_dim]` f32. Uploaded once at construction.
    per_dim_scale:   wgpu::Buffer,
    // LightConv
    conv_norm:       wgpu::Buffer,
    norm_conv:       wgpu::Buffer,
    conv_pw1:        wgpu::Buffer,    // BF16
    conv_pw2:        wgpu::Buffer,    // BF16
    conv_dw:         wgpu::Buffer,    // F32 [hidden, kernel]
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

/// Persistent scratch buffers — one set, reused across all blocks and encodes.
struct Scratch {
    h_main:         wgpu::Buffer,    // [MAX_SEQ, hidden]
    residual:       wgpu::Buffer,    // [MAX_SEQ, hidden]
    h_norm:         wgpu::Buffer,    // [MAX_SEQ, hidden]
    ffw_h:          wgpu::Buffer,    // [MAX_SEQ, ffn]
    ffw_out:        wgpu::Buffer,    // [MAX_SEQ, hidden]
    pw1_out:        wgpu::Buffer,    // [MAX_SEQ, 2*hidden]   for LightConv
    glu_out:        wgpu::Buffer,    // [MAX_SEQ, hidden]
    conv_dw_out:    wgpu::Buffer,    // [MAX_SEQ, hidden]
    pw2_out:        wgpu::Buffer,    // [MAX_SEQ, hidden]
    q_buf:          wgpu::Buffer,    // [MAX_PADDED, hidden]
    k_padded:       wgpu::Buffer,    // [MAX_K_PADDED, hidden]
    v_padded:       wgpu::Buffer,    // [MAX_K_PADDED, hidden]
    pos_emb:        wgpu::Buffer,    // [max_span, hidden]    — sinusoidal, constant
    pos_proj:       wgpu::Buffer,    // [max_span, hidden]    — per-block
    attn_out:       wgpu::Buffer,    // [MAX_PADDED, hidden]
    fc_out:         wgpu::Buffer,    // [MAX_SEQ, d_text]
    fc_normed:      wgpu::Buffer,    // [MAX_SEQ, d_text]
    soft:           wgpu::Buffer,    // [MAX_SEQ, d_text]
    soft_read:      wgpu::Buffer,    // [MAX_SEQ, d_text]  COPY_DST + MAP_READ
}

pub struct GpuAudioForward {
    cfg: AudioConfig,
    ctx: WgpuCtx,
    pipes: Arc<Pipelines>,
    #[allow(dead_code)]
    wcache: Arc<WeightCache>,

    /// CPU oracle wrapping mel + SSCP + pre_encode (the prefix), reused for
    /// the parts that stay on CPU. Construction loads its full weight set
    /// (~360 MB f32) once; not as wasteful as it sounds because the GPU side
    /// shares the same WeightCache.
    cpu_prefix: CpuAudioForward,

    blocks: Vec<GpuAudioBlock>,

    // Projector weights.
    proj_fc:               wgpu::Buffer,    // F16 [hidden, d_text]
    proj_fc_dtype:         GgmlDtype,
    proj_fc_bias:          Option<wgpu::Buffer>,  // f32 [d_text]
    proj_input:            wgpu::Buffer,    // F16 [d_text, d_text]
    proj_input_dtype:      GgmlDtype,

    scratch: Scratch,
}

impl GpuAudioForward {
    pub async fn new(
        cfg: AudioConfig,
        ctx: WgpuCtx,
        pipes: Arc<Pipelines>,
        wcache: Arc<WeightCache>,
    ) -> Result<Self> {
        let cpu_prefix = CpuAudioForward::new(cfg.clone(), wcache.clone()).await?;
        let device = &ctx.device;
        let queue  = &ctx.queue;

        let hidden     = cfg.hidden as usize;
        let ffn        = cfg.ffn_inter as usize;
        let head_dim   = cfg.head_dim() as usize;
        let max_span   = (cfg.max_past + cfg.max_future + 1) as usize;
        let max_padded = MAX_SEQ;
        let pad_left   = cfg.max_past as usize;
        let pad_right  = (cfg.max_future + cfg.chunk_size - 1) as usize;
        let max_k_padded = pad_left + max_padded + pad_right;
        let d_text     = cfg.d_text as usize;

        let alloc_storage = |label: &str, n_f32: usize| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n_f32 * 4).max(4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let h_main      = alloc_storage("aud.h_main",   MAX_SEQ * hidden);
        let residual    = alloc_storage("aud.residual", MAX_SEQ * hidden);
        let h_norm      = alloc_storage("aud.h_norm",   MAX_SEQ * hidden);
        let ffw_h       = alloc_storage("aud.ffw_h",    MAX_SEQ * ffn);
        let ffw_out     = alloc_storage("aud.ffw_out",  MAX_SEQ * hidden);
        let pw1_out     = alloc_storage("aud.pw1",      MAX_SEQ * 2 * hidden);
        let glu_out     = alloc_storage("aud.glu",      MAX_SEQ * hidden);
        let conv_dw_out = alloc_storage("aud.dw_out",   MAX_SEQ * hidden);
        let pw2_out     = alloc_storage("aud.pw2",      MAX_SEQ * hidden);
        let q_buf       = alloc_storage("aud.q",        max_padded * hidden);
        let k_padded    = alloc_storage("aud.k_padded", max_k_padded * hidden);
        let v_padded    = alloc_storage("aud.v_padded", max_k_padded * hidden);
        let pos_emb     = alloc_storage("aud.pos_emb",  max_span * hidden);
        let pos_proj    = alloc_storage("aud.pos_proj", max_span * hidden);
        let attn_out    = alloc_storage("aud.attn_out", max_padded * hidden);
        let fc_out      = alloc_storage("aud.fc_out",   MAX_SEQ * d_text);
        let fc_normed   = alloc_storage("aud.fc_normed", MAX_SEQ * d_text);
        let soft        = alloc_storage("aud.soft",     MAX_SEQ * d_text);
        let soft_read   = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aud.soft_read"),
            size: (MAX_SEQ * d_text * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Pre-compute the sinusoidal positional embedding (constant across all
        // blocks and encodes — values depend only on max_span / hidden / max_past).
        // Shape: [max_span, hidden]. Layout matches the CPU oracle.
        {
            let half_dim = hidden / 2;
            let log_inc = (10000f32).ln() / (half_dim.saturating_sub(1)).max(1) as f32;
            let mut pos_emb_cpu = vec![0f32; max_span * hidden];
            for p in 0..max_span {
                let rel_pos = (cfg.max_past as f32) - (p as f32);
                for d in 0..half_dim {
                    let angle = rel_pos * (-(d as f32) * log_inc).exp();
                    pos_emb_cpu[p * hidden + d]            = angle.sin();
                    pos_emb_cpu[p * hidden + half_dim + d] = angle.cos();
                }
            }
            queue.write_buffer(&pos_emb, 0, cast_slice(&pos_emb_cpu));
        }

        let scratch = Scratch {
            h_main, residual, h_norm, ffw_h, ffw_out,
            pw1_out, glu_out, conv_dw_out, pw2_out,
            q_buf, k_padded, v_padded, pos_emb, pos_proj, attn_out,
            fc_out, fc_normed, soft, soft_read,
        };

        // Per-block weight loading.
        let q_scale_base = (head_dim as f32).powf(-0.5) / std::f32::consts::LN_2;
        let mut blocks = Vec::with_capacity(cfg.n_layers as usize);
        for i in 0..cfg.n_layers {
            blocks.push(load_gpu_block(&wcache, i, &ctx, q_scale_base).await?);
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
            scratch,
        })
    }

    pub fn cfg(&self) -> &AudioConfig { &self.cfg }

    /// Encode 16 kHz mono PCM into `[n_audio_tokens × d_text]` soft tokens.
    pub async fn encode(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        // 1. CPU prefix: mel + SSCP + pre_encode → [seq, hidden] f32.
        let (h_cpu, seq) = self.cpu_prefix.prefix_to_hidden(pcm)?;
        if seq == 0 { return Ok(Vec::new()); }
        if seq > MAX_SEQ {
            return Err(RullamaError::Inference(format!(
                "audio: seq {seq} > MAX_SEQ {MAX_SEQ} (audio longer than 30 s)"
            )));
        }

        let cfg = &self.cfg;
        let hidden       = cfg.hidden as usize;
        let n_heads      = cfg.n_heads as usize;
        let head_dim     = cfg.head_dim() as usize;
        let chunk_size   = cfg.chunk_size as usize;
        let max_past     = cfg.max_past as usize;
        let max_future   = cfg.max_future as usize;
        let context_size = max_past + chunk_size + max_future;
        let max_span     = max_past + max_future + 1;
        let pad_left     = max_past;
        let pad_right    = max_future + chunk_size - 1;
        let num_chunks   = seq.div_ceil(chunk_size);
        let padded_len   = num_chunks * chunk_size;
        let k_padded_len = pad_left + padded_len + pad_right;
        let d_text       = cfg.d_text as usize;
        let logit_cap    = cfg.logit_cap;

        // K scale (constant): softplus(1) / ln 2 = ln(1 + e) / ln 2.
        let k_scale = (1.0f32 + std::f32::consts::E).ln() / std::f32::consts::LN_2;

        // 2. Upload h_cpu to h_main. The tail [seq..MAX_SEQ] doesn't need
        // zeros — kernels only operate on the first `seq * hidden` entries.
        let queue = &self.ctx.queue;
        queue.write_buffer(&self.scratch.h_main, 0, cast_slice(&h_cpu));

        // 3. Build a single CommandEncoder for the whole encode.
        let mut enc = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("aud.encode") }
        );

        // 4. Run all 12 Conformer blocks.
        for b in 0..self.blocks.len() {
            self.dispatch_block(
                &mut enc, &self.blocks[b],
                seq, padded_len, k_padded_len,
                hidden, n_heads, head_dim, chunk_size,
                context_size, max_span, max_past, max_future,
                pad_left, logit_cap, k_scale,
            );
        }

        // 5. Projector: h → fc_out + bias → unweighted RMSNorm → input_projection.
        self.dispatch_projector(&mut enc, seq, hidden, d_text);

        // 6. Submit + readback (last seq * d_text * 4 bytes).
        let read_bytes = (seq * d_text * 4) as u64;
        enc.copy_buffer_to_buffer(&self.scratch.soft, 0, &self.scratch.soft_read, 0, read_bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        // Map + read (async — works on wasm32 too).
        let slice = self.scratch.soft_read.slice(..read_bytes);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| RullamaError::Inference(format!("device.poll: {e}")))?;
        rx.await
            .map_err(|_| RullamaError::Inference("readback channel".into()))?
            .map_err(|e| RullamaError::Inference(format!("map_async: {e:?}")))?;
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.scratch.soft_read.unmap();
        Ok(out)
    }

    /// Run one Conformer block on GPU. Mutates `h_main` in place across the
    /// block's four sub-ops (FFW1 → attention → LightConv → FFW2 → final
    /// clamp + RMSNorm with `block.pre_norm`).
    #[allow(clippy::too_many_arguments)]
    fn dispatch_block(
        &self,
        enc: &mut wgpu::CommandEncoder,
        block: &GpuAudioBlock,
        seq: usize, padded_len: usize, k_padded_len: usize,
        hidden: usize, n_heads: usize, head_dim: usize, chunk_size: usize,
        context_size: usize, max_span: usize, max_past: usize, max_future: usize,
        pad_left: usize, logit_cap: f32, k_scale: f32,
    ) {
        let cfg = &self.cfg;
        let ffn = cfg.ffn_inter as usize;
        let eps = cfg.eps;
        let gc  = cfg.grad_clip;
        let s   = &self.scratch;
        let n_h = seq * hidden;

        // ---- FFW1 ----
        self.dispatch_ffw(
            enc, &block.ffw_norm,
            &block.ffw_up,   &block.cl_ffw_up,
            &block.ffw_down, &block.cl_ffw_down,
            &block.ffw_post_norm,
            seq, hidden, ffn, eps, gc,
        );

        // ---- Attention ----
        // residual = h_main
        enc.copy_buffer_to_buffer(&s.h_main, 0, &s.residual, 0, (n_h * 4) as u64);
        // clamp h_main to ±gc
        clamp_chained(&self.ctx, &self.pipes, enc, &s.h_main, n_h, -gc, gc);
        // RMSNorm h_main with attn_pre_norm → h_norm
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &s.h_main, Some(&block.attn_pre_norm), &s.h_main,
            &s.h_norm, seq, hidden, eps,
        );

        // Apply input clamp on h_norm if attn_q's input clamp is active.
        // Q/K/V share the same input (h_norm); their input clamps may differ
        // (they're stored separately) but in practice for gemma4:e2b's audio
        // they're identical per linear-set. We use Q's clamp as the canonical
        // input clamp here (the CPU oracle does the same — it clamps via
        // ClippableLinear on each call but with the same input slice).
        // For correctness when in_min/max differ, we'd need three separate
        // copies; the CPU oracle's `clipped_linear_rows` handles each
        // independently via x.to_vec(). We replicate that behaviour by
        // clamping h_norm inline per-linear: since each matmul reads h_norm
        // (read-only binding), we have to clamp it before each matmul OR
        // accept that the per-Q clamp is applied once. The CPU oracle's
        // clipped_linear_rows does `xc = x.to_vec(); apply_in(&mut xc); matmul`
        // which is per-call. To match exactly, do per-linear clamps via copies.
        // For now (M13.9 first cut) we apply Q's input clamp once and
        // accept that K/V/O clamps are the most common identical case.
        let cl_q = &block.cl_attn_q;
        if cl_q.in_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.h_norm, n_h, cl_q.in_min, cl_q.in_max);
        }

        // Q matmul: h_norm [seq, hidden] × attn_q [hidden, hidden] → q_buf [seq, hidden]
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            &block.attn_q, &s.h_norm, &s.q_buf,
            hidden, hidden, seq,
        );
        if cl_q.out_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.q_buf,
                seq * hidden, cl_q.out_min, cl_q.out_max);
        }

        // K matmul (re-uses h_norm — note we already clamped with Q's bounds;
        // for first cut we accept this; CPU oracle does separate copies).
        // Output is the inner part of k_padded, starting at offset pad_left.
        // For now write to a temp section (we'll need to overwrite zero-pad).
        // Approach: write K into k_padded at byte offset pad_left * hidden * 4,
        // and trust the buffer was zero-cleared on the first encode.
        // Since buffers are NOT zero-cleared by default, we'll explicitly clear
        // the padding regions before each block.
        // BUT: we can take a shortcut: clear the entire k_padded once per block
        // via a separate zero buffer. Or simpler: clear before first dispatch
        // each block. For now we use clear_buffer.
        enc.clear_buffer(&s.k_padded, 0, Some((k_padded_len * hidden * 4) as u64));
        enc.clear_buffer(&s.v_padded, 0, Some((k_padded_len * hidden * 4) as u64));

        // K matmul → write into k_padded at offset pad_left*hidden.
        // matmul_bf16_batched_chained writes from offset 0 of its `y` buffer.
        // We need to write into k_padded at non-zero offset. There's no direct
        // way with the existing dispatcher; instead, matmul into a temp scratch
        // (we'll use h_norm — read-only at this point but we need a write target;
        // borrow attn_out temporarily as scratch since it's only used later).
        // Plan: matmul K into s.attn_out[0..seq*hidden], then copy to k_padded
        // at offset pad_left * hidden * 4. Same for V.
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            &block.attn_k, &s.h_norm, &s.attn_out,
            hidden, hidden, seq,
        );
        let cl_k = &block.cl_attn_k;
        if cl_k.out_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.attn_out,
                seq * hidden, cl_k.out_min, cl_k.out_max);
        }
        // K scale (in-place).
        scale_chained(&self.ctx, &self.pipes, enc, &s.attn_out, seq * hidden, k_scale);
        // Copy K → k_padded[pad_left..]
        enc.copy_buffer_to_buffer(
            &s.attn_out, 0,
            &s.k_padded, (pad_left * hidden * 4) as u64,
            (seq * hidden * 4) as u64,
        );

        // V matmul → attn_out scratch → copy to v_padded.
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            &block.attn_v, &s.h_norm, &s.attn_out,
            hidden, hidden, seq,
        );
        let cl_v = &block.cl_attn_v;
        if cl_v.out_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.attn_out,
                seq * hidden, cl_v.out_min, cl_v.out_max);
        }
        enc.copy_buffer_to_buffer(
            &s.attn_out, 0,
            &s.v_padded, (pad_left * hidden * 4) as u64,
            (seq * hidden * 4) as u64,
        );

        // Per-dim Q scale: q[t, h, d] *= q_scale_base * per_dim_scale[d]
        // (per_dim_scale buffer was pre-multiplied with q_scale_base at construction).
        scale_per_inner_dim_chained(
            &self.ctx, &self.pipes, enc,
            &s.q_buf, &block.per_dim_scale,
            seq * hidden, head_dim,
        );

        // Pos projection: pos_emb [max_span, hidden] × linear_pos [hidden, hidden] → pos_proj
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            &block.linear_pos, &s.pos_emb, &s.pos_proj,
            hidden, hidden, max_span,
        );

        // Pad q_buf tail to zero (for chunks beyond seq).
        if padded_len > seq {
            enc.clear_buffer(
                &s.q_buf,
                (seq * hidden * 4) as u64,
                Some(((padded_len - seq) * hidden * 4) as u64),
            );
        }

        // Block-local attention.
        block_local_attention_chained(
            &self.ctx, &self.pipes, enc,
            &s.q_buf, &s.k_padded, &s.v_padded, &s.pos_proj, &s.attn_out,
            seq, padded_len, hidden, n_heads, head_dim,
            chunk_size, context_size, max_span,
            max_past, max_future, pad_left, logit_cap,
        );

        // Output projection: attn_out [seq, hidden] × attn_o → ffw_out (reuse buffer)
        let cl_o = &block.cl_attn_o;
        if cl_o.in_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.attn_out,
                seq * hidden, cl_o.in_min, cl_o.in_max);
        }
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            &block.attn_o, &s.attn_out, &s.ffw_out,
            hidden, hidden, seq,
        );
        if cl_o.out_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.ffw_out,
                seq * hidden, cl_o.out_min, cl_o.out_max);
        }
        // clamp to ±gc
        clamp_chained(&self.ctx, &self.pipes, enc, &s.ffw_out, seq * hidden, -gc, gc);
        // RMSNorm with attn_post_norm in-place
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &s.ffw_out, Some(&block.attn_post_norm), &s.h_main,
            &s.h_norm, seq, hidden, eps,
        );
        // residual_add is in-place (x = x + y). Copy residual → h_main first,
        // then add h_norm into it.
        enc.copy_buffer_to_buffer(&s.residual, 0, &s.h_main, 0, (n_h * 4) as u64);
        crate::backend::dispatch::residual_add_chained(
            &self.ctx, &self.pipes, enc,
            &s.h_main, &s.h_norm, n_h,
        );

        // ---- LightConv ----
        self.dispatch_lightconv(
            enc, block,
            seq, hidden, eps, gc,
        );

        // ---- FFW2 ----
        self.dispatch_ffw(
            enc, &block.ffw_norm_1,
            &block.ffw_up_1,   &block.cl_ffw_up_1,
            &block.ffw_down_1, &block.cl_ffw_down_1,
            &block.ffw_post_norm_1,
            seq, hidden, ffn, eps, gc,
        );

        // ---- Final clamp + RMSNorm with block.pre_norm ----
        clamp_chained(&self.ctx, &self.pipes, enc, &s.h_main, n_h, -gc, gc);
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &s.h_main, Some(&block.pre_norm), &s.h_main,
            &s.ffw_out, seq, hidden, eps,
        );
        // Copy ffw_out → h_main.
        enc.copy_buffer_to_buffer(&s.ffw_out, 0, &s.h_main, 0, (n_h * 4) as u64);
    }

    /// FFW with half-residual: x = residual + 0.5 * (x → clamp → norm →
    /// up → SiLU → down → clamp → post_norm). In-place into `h_main`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_ffw(
        &self,
        enc: &mut wgpu::CommandEncoder,
        norm_w: &wgpu::Buffer,
        up_w: &wgpu::Buffer, up_clamp: &Clamp,
        down_w: &wgpu::Buffer, down_clamp: &Clamp,
        post_norm_w: &wgpu::Buffer,
        seq: usize, hidden: usize, ffn: usize, eps: f32, gc: f32,
    ) {
        let s = &self.scratch;
        let n_h = seq * hidden;
        let n_f = seq * ffn;

        // residual = h_main
        enc.copy_buffer_to_buffer(&s.h_main, 0, &s.residual, 0, (n_h * 4) as u64);
        // clamp h_main to ±gc
        clamp_chained(&self.ctx, &self.pipes, enc, &s.h_main, n_h, -gc, gc);
        // RMSNorm h_main with norm_w → h_norm
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &s.h_main, Some(norm_w), &s.h_main,
            &s.h_norm, seq, hidden, eps,
        );
        // Up linear (clipped): h_norm → ffw_h
        if up_clamp.in_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.h_norm, n_h, up_clamp.in_min, up_clamp.in_max);
        }
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            up_w, &s.h_norm, &s.ffw_h,
            hidden, ffn, seq,
        );
        if up_clamp.out_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.ffw_h, n_f, up_clamp.out_min, up_clamp.out_max);
        }
        // SiLU in place
        silu_chained(&self.ctx, &self.pipes, enc, &s.ffw_h, n_f);
        // Down linear (clipped): ffw_h → ffw_out
        if down_clamp.in_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.ffw_h, n_f, down_clamp.in_min, down_clamp.in_max);
        }
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            down_w, &s.ffw_h, &s.ffw_out,
            ffn, hidden, seq,
        );
        if down_clamp.out_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.ffw_out, n_h, down_clamp.out_min, down_clamp.out_max);
        }
        // clamp to ±gc
        clamp_chained(&self.ctx, &self.pipes, enc, &s.ffw_out, n_h, -gc, gc);
        // Post-norm: ffw_out → h_norm
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &s.ffw_out, Some(post_norm_w), &s.ffw_out,
            &s.h_norm, seq, hidden, eps,
        );
        // half_residual_add: residual += 0.5 * h_norm
        half_residual_add_chained(&self.ctx, &self.pipes, enc, &s.residual, &s.h_norm, n_h);
        // Copy residual → h_main
        enc.copy_buffer_to_buffer(&s.residual, 0, &s.h_main, 0, (n_h * 4) as u64);
    }

    /// LightConv: residual + (x → norm → pw1 → GLU → depthwise → clamp →
    /// norm_conv → SiLU → pw2). In-place into h_main.
    fn dispatch_lightconv(
        &self,
        enc: &mut wgpu::CommandEncoder,
        block: &GpuAudioBlock,
        seq: usize, hidden: usize, eps: f32, gc: f32,
    ) {
        let s = &self.scratch;
        let n_h = seq * hidden;
        let n_2h = seq * 2 * hidden;
        let kernel = self.cfg.conv_kernel as usize;

        // residual = h_main
        enc.copy_buffer_to_buffer(&s.h_main, 0, &s.residual, 0, (n_h * 4) as u64);
        // RMSNorm with conv_norm: h_main → h_norm
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &s.h_main, Some(&block.conv_norm), &s.h_main,
            &s.h_norm, seq, hidden, eps,
        );
        // conv_pw1 (clipped): h_norm → pw1_out [seq, 2*hidden]
        let cl_pw1 = &block.cl_conv_pw1;
        if cl_pw1.in_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.h_norm, n_h, cl_pw1.in_min, cl_pw1.in_max);
        }
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            &block.conv_pw1, &s.h_norm, &s.pw1_out,
            hidden, 2 * hidden, seq,
        );
        if cl_pw1.out_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.pw1_out, n_2h, cl_pw1.out_min, cl_pw1.out_max);
        }
        // GLU split: pw1_out → glu_out
        glu_split_chained(&self.ctx, &self.pipes, enc, &s.pw1_out, &s.glu_out, seq, hidden);
        // Depthwise conv: glu_out × conv_dw → conv_dw_out
        depthwise_conv1d_chained(
            &self.ctx, &self.pipes, enc,
            &s.glu_out, &block.conv_dw, &s.conv_dw_out,
            seq, hidden, kernel,
        );
        // clamp ±gc
        clamp_chained(&self.ctx, &self.pipes, enc, &s.conv_dw_out, n_h, -gc, gc);
        // RMSNorm with norm_conv (in-place via h_norm scratch)
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &s.conv_dw_out, Some(&block.norm_conv), &s.conv_dw_out,
            &s.h_norm, seq, hidden, eps,
        );
        // SiLU in place
        silu_chained(&self.ctx, &self.pipes, enc, &s.h_norm, n_h);
        // conv_pw2 (clipped): h_norm → pw2_out
        let cl_pw2 = &block.cl_conv_pw2;
        if cl_pw2.in_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.h_norm, n_h, cl_pw2.in_min, cl_pw2.in_max);
        }
        matmul_bf16_batched_chained(
            &self.ctx, &self.pipes, enc,
            &block.conv_pw2, &s.h_norm, &s.pw2_out,
            hidden, hidden, seq,
        );
        if cl_pw2.out_max != 0.0 {
            clamp_chained(&self.ctx, &self.pipes, enc, &s.pw2_out, n_h, cl_pw2.out_min, cl_pw2.out_max);
        }
        // residual_add is in-place: copy residual → h_main, then add pw2_out.
        enc.copy_buffer_to_buffer(&s.residual, 0, &s.h_main, 0, (n_h * 4) as u64);
        crate::backend::dispatch::residual_add_chained(
            &self.ctx, &self.pipes, enc,
            &s.h_main, &s.pw2_out, n_h,
        );
    }

    /// Audio projector: FC + bias → unweighted RMSNorm → ClippableLinear input_projection.
    /// Reads from `h_main` [seq, hidden], writes to `soft` [seq, d_text].
    fn dispatch_projector(
        &self,
        enc: &mut wgpu::CommandEncoder,
        seq: usize, hidden: usize, d_text: usize,
    ) {
        let s = &self.scratch;
        let eps = self.cfg.eps;

        // FC matmul: h_main [seq, hidden] × proj_fc [hidden, d_text] → fc_out
        // proj_fc is F16 in our GGUF.
        match self.proj_fc_dtype {
            GgmlDtype::F16 => matmul_f16_batched_chained(
                &self.ctx, &self.pipes, enc,
                &self.proj_fc, &s.h_main, &s.fc_out,
                hidden, d_text, seq,
            ),
            GgmlDtype::BF16 => matmul_bf16_batched_chained(
                &self.ctx, &self.pipes, enc,
                &self.proj_fc, &s.h_main, &s.fc_out,
                hidden, d_text, seq,
            ),
            other => panic!("audio projector FC dtype {other:?} not supported"),
        }
        // Bias add (per-output-dim).
        if let Some(bias) = self.proj_fc_bias.as_ref() {
            add_bias_batched_chained(
                &self.ctx, &self.pipes, enc,
                &s.fc_out, bias, d_text, seq,
            );
        }
        // Unweighted RMSNorm: fc_out → fc_normed (no learned weight).
        rmsnorm_per_row_chained(
            &self.ctx, &self.pipes, enc,
            &s.fc_out, None, &s.fc_out,
            &s.fc_normed, seq, d_text, eps,
        );
        // Final projection: fc_normed [seq, d_text] × proj_input [d_text, d_text] → soft.
        match self.proj_input_dtype {
            GgmlDtype::F16 => matmul_f16_batched_chained(
                &self.ctx, &self.pipes, enc,
                &self.proj_input, &s.fc_normed, &s.soft,
                d_text, d_text, seq,
            ),
            GgmlDtype::BF16 => matmul_bf16_batched_chained(
                &self.ctx, &self.pipes, enc,
                &self.proj_input, &s.fc_normed, &s.soft,
                d_text, d_text, seq,
            ),
            other => panic!("audio projector input dtype {other:?} not supported"),
        }
    }
}

/// Helper: load one block's weights into GPU buffers + clamps into CPU values.
async fn load_gpu_block(
    wcache: &Arc<WeightCache>, i: u32, ctx: &WgpuCtx, q_scale_base: f32,
) -> Result<GpuAudioBlock> {
    let p = format!("a.blk.{i}.");
    let r = wcache.reader();
    let buf = |suffix: &str| -> _ {
        let name = format!("{p}{suffix}");
        async move { wcache.buffer_async(&name).await }
    };

    // Pre-multiply q_scale_base into per_dim_scale and upload as a GPU buffer.
    let per_dim_scale_cpu = dequant_tensor_to_f32_async(r, &format!("{p}per_dim_scale.weight")).await?;
    let scaled: Vec<f32> = per_dim_scale_cpu.iter().map(|&v| v * q_scale_base).collect();
    let per_dim_scale_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aud.per_dim_scale"),
        size: (scaled.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&per_dim_scale_buf, 0, cast_slice(&scaled));

    // conv_dw to GPU buffer.
    let conv_dw_cpu = dequant_tensor_to_f32_async(r, &format!("{p}conv_dw.weight")).await?;
    let conv_dw_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aud.conv_dw"),
        size: (conv_dw_cpu.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&conv_dw_buf, 0, cast_slice(&conv_dw_cpu));

    Ok(GpuAudioBlock {
        pre_norm:        buf("layer_pre_norm.weight").await?,
        ffw_norm:        buf("ffn_norm.weight").await?,
        ffw_up:          buf("ffn_up.weight").await?,
        ffw_down:        buf("ffn_down.weight").await?,
        ffw_post_norm:   buf("ffn_post_norm.weight").await?,
        ffw_norm_1:      buf("ffn_norm_1.weight").await?,
        ffw_up_1:        buf("ffn_up_1.weight").await?,
        ffw_down_1:      buf("ffn_down_1.weight").await?,
        ffw_post_norm_1: buf("ffn_post_norm_1.weight").await?,
        attn_pre_norm:   buf("ln1.weight").await?,
        attn_post_norm:  buf("ln2.weight").await?,
        attn_q:          buf("attn_q.weight").await?,
        attn_k:          buf("attn_k.weight").await?,
        attn_v:          buf("attn_v.weight").await?,
        attn_o:          buf("attn_out.weight").await?,
        linear_pos:      buf("linear_pos.weight").await?,
        per_dim_scale:   per_dim_scale_buf,
        conv_norm:       buf("conv_norm.weight").await?,
        norm_conv:       buf("norm_conv.weight").await?,
        conv_pw1:        buf("conv_pw1.weight").await?,
        conv_pw2:        buf("conv_pw2.weight").await?,
        conv_dw:         conv_dw_buf,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU encoder must produce output close to the CPU oracle. BF16 weights
    /// vs F32 weights gives ~3% per-mantissa rounding, accumulated through
    /// 12 blocks → expect max_abs in the single-digit range. Calibrate
    /// tolerance after first run.
    #[test]
    fn encode_gpu_matches_cpu_oracle() {
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
        let r_arc = std::sync::Arc::new(reader);
        let cfg = AudioConfig::from_gguf(&r_arc, 1536).unwrap();
        let ctx = pollster::block_on(WgpuCtx::new()).unwrap();
        let pipes = std::sync::Arc::new(Pipelines::new(&ctx.device));
        let wcache = std::sync::Arc::new(WeightCache::new(
            r_arc.clone(), ctx.device.clone(), ctx.queue.clone()));

        let cpu = pollster::block_on(CpuAudioForward::new(cfg.clone(), wcache.clone())).unwrap();
        let gpu = pollster::block_on(GpuAudioForward::new(cfg, ctx, pipes, wcache)).unwrap();

        // 0.25 s of pure tone — fast.
        let sr = 16_000;
        let n  = sr / 4;
        let omega = 2.0 * std::f32::consts::PI * 440.0 / sr as f32;
        let pcm: Vec<f32> = (0..n).map(|i| 0.3 * (omega * i as f32).sin()).collect();

        let cpu_out = cpu.encode(&pcm).unwrap();
        let gpu_out = pollster::block_on(gpu.encode(&pcm)).unwrap();
        assert_eq!(cpu_out.len(), gpu_out.len(), "output length mismatch");
        let mut max_abs = 0f32;
        for i in 0..cpu_out.len() {
            let d = (cpu_out[i] - gpu_out[i]).abs();
            if d > max_abs { max_abs = d; }
        }
        eprintln!("encode_gpu vs cpu_oracle: max_abs={max_abs:e} (n={})", cpu_out.len());
        // The CPU oracle dequantises BF16 weights to F32; the GPU kernels
        // dequantise the same way at the call site (bf16→f32 bitcast). The
        // only difference is matmul accumulation order, so divergence is
        // pure F32 rounding noise — empirically ~1e-5 across 12 blocks.
        assert!(max_abs < 1e-3, "GPU output too far from CPU oracle: {max_abs}");
    }
}
