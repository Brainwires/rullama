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
/// (`step_capture`) writes activations at per-position offsets, so a
/// single `[d_model]` buffer overruns. `seq_len` must be ≥ the
/// longest sequence the caller will forward through the captured
/// path.
pub struct RomeCapture {
    ctx: Arc<WgpuCtx>,
    /// `d_model` (= width of the residual stream). Reserved for
    /// Phase 1.2 — v* lives in this dimension so the future
    /// `read_ffn_out` (or backward-gradient readback) will need
    /// `d_model`-sized staging buffers.
    #[allow(dead_code)]
    cfg_d_model: u32,
    /// Per-layer `ffn_inter` (= d_ffn) for sized readback of ffn_act.
    /// Differs across layers in Gemma 4 e2b.
    cfg_ffn_inter: Vec<u32>,
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

        let cfg_ffn_inter: Vec<u32> = (0..cfg.n_layers).map(|i| cfg.ffn(i)).collect();
        Self {
            ctx: Arc::clone(ctx),
            cfg_d_model: cfg.d_model,
            cfg_ffn_inter,
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

    /// Read back `ffn_act[target_layer]` at `position` as `[d_ffn]`
    /// f32. **This is ROME's k\*** — the post-GEGLU activation that
    /// is the INPUT to `ffn_down`. Required for a rank-1 LoRA-style
    /// update on `ffn_down.weight` (shape `[d_model × d_ffn]`) where
    /// the rank-1 factor A must be `[d_ffn]`-shaped to compose.
    ///
    /// (We previously read `norm_x_ffn` of shape `[d_model]` — wrong
    /// shape for the rank-1 update on `ffn_down`. The reference
    /// ROME implementation also extracts post-activation, not
    /// pre-MLP-input.)
    ///
    /// Capture buffer is seq-shaped (`[seq_len × ffn_inter]`);
    /// extract slice at byte offset `position × ffn_inter × 4`.
    pub async fn read_ffn_act(
        &self,
        target_layer: u32,
        position: u32,
    ) -> Result<Vec<f32>> {
        let layer = target_layer as usize;
        if layer >= self.layers.len() {
            return Err(RullamaError::Inference(format!(
                "read_ffn_act: layer {layer} out of range (have {})",
                self.layers.len()
            )));
        }
        if position >= self.seq_len {
            return Err(RullamaError::Inference(format!(
                "read_ffn_act: position {position} >= seq_len {}",
                self.seq_len
            )));
        }
        let ffn_inter = self.cfg_ffn_inter[layer] as u64;
        let src = &self.layers[layer].ffn_act;
        let elt_bytes = ffn_inter * 4;
        let src_offset = (position as u64) * elt_bytes;
        let bytes = elt_bytes;
        let staging = self.ctx.device.create_buffer(&BufferDescriptor {
            label: Some("rome.staging.ffn_act"),
            size: bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rome.read_ffn_act"),
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

// ─────────────────────────────────────────────────────────────────────
// Phase 1.2 — v* computation (planned, not yet implemented).
//
// Per the ROME paper:
//   v* = (W_down @ k*) + δ
// where δ is found by minimizing CE loss against the target token
// while injecting δ into the residual stream at layer L's output
// position. Iterated Adam for ~20 steps with lr ≈ 0.5.
//
// IMPLEMENTATION STRATEGY (chose during Phase 1.1 code review,
// trades full-paper fidelity for reuse of existing backward path):
//
// Instead of perturbing the residual at layer L and re-forwarding
// (which requires a new partial-forward code path not in the engine),
// use a FIRST-ORDER approximation: compute the gradient
// `∂loss/∂hidden_at_layer_L+1_input` via the existing
// `Forward::backward_step` with all LoRA slots/grads = None and
// `backward_layer_floor = target_layer + 1`. The residual-stream
// gradient at that point equals `∂loss/∂(attn_residual + ffn_residual)`
// at layer L = `∂loss/∂ffn_out[L]` (the attn_residual doesn't depend
// on what we're optimizing).
//
// Then v* = current_ffn_out[L, last_pos] + lr * (−gradient).
//
// This is single-step; iterative gradient descent would require
// re-running forward with the updated δ injected, which is the
// partial-forward path we're explicitly avoiding for v1.
//
// REQUIRED WORK (~3-4 days):
//   1. BackwardScratch allocator — owns ~40 wgpu::Buffer fields
//      mirroring BackwardScratchView (forward_chained.rs:2484-2570).
//      Today only TrainingScratch in rullama-finetune does this; we
//      can't depend on that crate from rullama (cycle). Either:
//        a. Add `BackwardScratch::new()` helper in rullama and have
//           TrainingScratch use it (refactor — cleanest)
//        b. Duplicate the allocation in rome.rs (faster, accepts
//           future drift)
//   2. EmptyLoraSlots / EmptyLoraGrads helpers — `Vec<LayerLoraSlots>`
//      with all fields `None`, length = n_layers.
//   3. `RomeCapture::read_ffn_out(layer, position) -> Vec<f32>` —
//      same pattern as read_ffn_act but on the ffn_out buffer
//      (shape `[d_model]`).
//   4. `Forward::backward_to_layer(target_layer, target_token_id,
//      capture, scratch) -> Result<Vec<f32>>` — orchestration that
//      calls backward_step with backward_layer_floor=target_layer+1
//      then reads back scratch.d_hidden as the v* gradient direction.
//   5. `Model::compute_rome_gradient_native(prompt_tokens,
//      target_layer, target_token_id) -> Result<Vec<f32>>` — public
//      API wiring k* extraction + scratch alloc + backward.
//
// DEFER: full iterative v* (multi-step δ optimization). The 1-step
// approximation is a real first-order ROME variant the paper
// discusses in Appendix C.
// ─────────────────────────────────────────────────────────────────────

