//! ROME (Rank-One Model Editing) infrastructure for surgical fact
//! substitution. Phase 1.1 of the ROME/MEMIT plan
//! (`.claude/plans/write-this-up-formally-delegated-sun.md`).
//!
//! This module owns the GPU buffer infrastructure ROME needs that
//! sits on top of the existing inference path:
//!
//!   • `RomeCapture` — per-layer single-position activation buffers
//!     sized to hold what `Forward::step_capture` writes. Differs
//!     from `rullama_finetune::scratch::LayerActivations` only in
//!     that ROME doesn't need the sequence-dimensional storage that
//!     training uses for the PerPosition backward sweep.
//!
//!   • `RomeCapture::as_captures` — produces a `Vec<LayerCaptureBuffers>`
//!     view that can be passed directly to `Forward::step_capture`.
//!
//!   • `RomeCapture::read_norm_x_ffn` — async readback of the MLP-input
//!     activation (k* in the ROME formulation) at a chosen layer.
//!
//! The full ROME algorithm (k* extraction + v* gradient descent + rank-1
//! safetensors serialization) is implemented in `crate::api::Model`
//! using these primitives. See the plan file for the math.

use std::sync::Arc;

use futures_channel::oneshot;
use wgpu::{Buffer, BufferDescriptor, BufferUsages};

use crate::backend::WgpuCtx;
use crate::error::{Result, RullamaError};
use crate::model::config::Gemma4Config;
use crate::reference::forward_chained::LayerCaptureBuffers;

/// Per-layer single-position capture buffers. Mirrors the shape
/// requirements of `LayerCaptureBuffers` exactly so the existing
/// `Forward::step_capture` path can write into them.
///
/// Sized for ONE token position (the subject's last token). ROME
/// doesn't need sequence-shaped captures the way per-position
/// training does — we only care about the last-position MLP-input
/// vector for k* extraction and (later) for v* gradient descent.
struct RomeLayerBuffers {
    hidden_in: Buffer,
    norm_x_attn: Buffer,
    q_pre_norm: Buffer,
    q_post_rope: Buffer,
    k_pre_norm: Buffer,
    v_pre_norm: Buffer,
    attn_out: Buffer,
    attn_proj: Buffer,
    pre_ffn_rms: Buffer,
    norm_x_ffn: Buffer,
    ffn_gate: Buffer,
    ffn_up: Buffer,
    ffn_act: Buffer,
    ffn_out: Buffer,
    ple_state: Buffer,
    ple_act: Buffer,
    ple_proj: Buffer,
}

/// Collection of per-layer capture buffers for a ROME forward pass.
/// Allocate once with [`RomeCapture::new`], then convert to
/// `&[LayerCaptureBuffers]` via [`RomeCapture::as_captures`] each
/// time the caller invokes `Forward::step_capture`.
///
/// Buffers are sized to `seq_len × per_position`. The Forward path
/// (`step_capture`) writes activations at per-position offsets
/// (`copy at byte_offset = position * d_model * 4`), so a single
/// `[d_model]` buffer overruns. `seq_len` must be ≥ the longest
/// sequence the caller will forward through the captured path.
pub struct RomeCapture {
    ctx: Arc<WgpuCtx>,
    cfg_d_model: u32,
    seq_len: u32,
    layers: Vec<RomeLayerBuffers>,
}

impl RomeCapture {
    /// Allocate per-layer capture buffers sized for sequences up to
    /// `seq_len` tokens. Each buffer is
    /// `STORAGE | COPY_DST | COPY_SRC` so the kernels can write into
    /// it and we can `copy_buffer_to_buffer` the contents to a
    /// `MAP_READ` staging buffer for CPU readback.
    ///
    /// Memory cost is roughly `seq_len × (d_model + ffn_inter + ...)
    /// × 4 × n_layers` bytes. For Gemma 4 e2b at seq=64 that's about
    /// 30-50 MB total — small relative to the model.
    pub fn new(ctx: &Arc<WgpuCtx>, cfg: &Gemma4Config, seq_len: u32) -> Self {
        let device = &ctx.device;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

        let make = |label: &'static str, elems: u64| -> Buffer {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: (elems * 4).max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        let d_model = cfg.d_model as u64;
        let seq = seq_len as u64;
        let n_heads = cfg.n_heads as u64;
        let head_dim_max = cfg.head_dim_global.max(cfg.head_dim_swa) as u64;
        let n_kv_max = cfg.n_kv_heads_global.max(cfg.n_kv_heads_swa) as u64;
        let ple_dim = if cfg.has_ple() { cfg.ple_dim as u64 } else { 0 };
        let ple_d = if ple_dim > 0 { d_model } else { 0 };

        let layers: Vec<RomeLayerBuffers> = (0..cfg.n_layers)
            .map(|li| {
                let ffn_inter = cfg.ffn(li) as u64;
                RomeLayerBuffers {
                    hidden_in: make("rome.hidden_in", d_model * seq),
                    norm_x_attn: make("rome.norm_x_attn", d_model * seq),
                    q_pre_norm: make("rome.q_pre_norm", n_heads * head_dim_max * seq),
                    q_post_rope: make("rome.q_post_rope", n_heads * head_dim_max * seq),
                    k_pre_norm: make("rome.k_pre_norm", n_kv_max * head_dim_max * seq),
                    v_pre_norm: make("rome.v_pre_norm", n_kv_max * head_dim_max * seq),
                    attn_out: make("rome.attn_out", n_heads * head_dim_max * seq),
                    attn_proj: make("rome.attn_proj", d_model * seq),
                    pre_ffn_rms: make("rome.pre_ffn_rms", d_model * seq),
                    norm_x_ffn: make("rome.norm_x_ffn", d_model * seq),
                    ffn_gate: make("rome.ffn_gate", ffn_inter * seq),
                    ffn_up: make("rome.ffn_up", ffn_inter * seq),
                    ffn_act: make("rome.ffn_act", ffn_inter * seq),
                    ffn_out: make("rome.ffn_out", d_model * seq),
                    ple_state: make("rome.ple_state", ple_dim * seq),
                    ple_act: make("rome.ple_act", ple_dim * seq),
                    ple_proj: make("rome.ple_proj", ple_d * seq),
                }
            })
            .collect();

        Self {
            ctx: Arc::clone(ctx),
            cfg_d_model: cfg.d_model,
            seq_len,
            layers,
        }
    }

    /// View suitable for `Forward::step_capture(&captures, ...)`. The
    /// returned Vec borrows from `self`; valid as long as the caller
    /// holds it (the `step_capture` call itself is short-lived).
    pub fn as_captures(&self) -> Vec<LayerCaptureBuffers<'_>> {
        self.layers
            .iter()
            .map(|l| LayerCaptureBuffers {
                hidden_in: &l.hidden_in,
                norm_x_attn: &l.norm_x_attn,
                q_pre_norm: &l.q_pre_norm,
                q_post_rope: &l.q_post_rope,
                k_pre_norm: &l.k_pre_norm,
                v_pre_norm: &l.v_pre_norm,
                attn_out: &l.attn_out,
                attn_proj: &l.attn_proj,
                pre_ffn_rms: &l.pre_ffn_rms,
                norm_x_ffn: &l.norm_x_ffn,
                ffn_gate: &l.ffn_gate,
                ffn_up: &l.ffn_up,
                ffn_act: &l.ffn_act,
                ffn_out: &l.ffn_out,
                ple_state: &l.ple_state,
                ple_act: &l.ple_act,
                ple_proj: &l.ple_proj,
            })
            .collect()
    }

    /// Read back `norm_x_ffn[target_layer]` at `position` as
    /// `[d_model]` f32. This is ROME's **k\*** — the post-RMSNorm
    /// pre-FFN activation at the subject's last token.
    ///
    /// The capture buffer is seq-shaped (`[seq_len × d_model]`); we
    /// extract the slice at byte offset `position × d_model × 4`.
    ///
    /// Must be called AFTER a `Forward::step_capture` for that
    /// position. Calling before any capture yields zeros.
    pub async fn read_norm_x_ffn(
        &self,
        target_layer: u32,
        position: u32,
    ) -> Result<Vec<f32>> {
        let layer = target_layer as usize;
        if layer >= self.layers.len() {
            return Err(RullamaError::Inference(format!(
                "read_norm_x_ffn: layer {layer} out of range (have {})",
                self.layers.len()
            )));
        }
        if position >= self.seq_len {
            return Err(RullamaError::Inference(format!(
                "read_norm_x_ffn: position {position} >= seq_len {}",
                self.seq_len
            )));
        }
        let src = &self.layers[layer].norm_x_ffn;
        let d_model_bytes = (self.cfg_d_model as u64) * 4;
        let src_offset = (position as u64) * d_model_bytes;
        let bytes = d_model_bytes;
        let staging = self.ctx.device.create_buffer(&BufferDescriptor {
            label: Some("rome.staging.norm_x_ffn"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rome.read_norm_x_ffn"),
            });
        enc.copy_buffer_to_buffer(src, src_offset, &staging, 0, bytes);
        self.ctx.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| RullamaError::Inference(format!("{e:?}")))?;
        receiver
            .await
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?
            .map_err(|e| RullamaError::BufferMap(format!("{e}")))?;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(v)
    }
}
